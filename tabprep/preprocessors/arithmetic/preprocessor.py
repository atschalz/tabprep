from tabprep.preprocessors.arithmetic.combinations import *
from tabprep.preprocessors.arithmetic.filtering import *
from tabprep.preprocessors.arithmetic.memory import *
from tabprep.preprocessors.base import BasePreprocessor
import re

from time import perf_counter
from contextlib import contextmanager

class TimerLog:
    def __init__(self):
        self.times = {}

    @contextmanager
    def block(self, name):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            self.times[name] = self.times.get(name, 0) + dt
            print(f"[{name}] {dt:.3f}s")

    def summary(self):
        print("\n--- Timing Summary (in order) ---")
        for name, total in self.times.items():
            print(f"{name:<20} {total:.3f}s")
        return dict(self.times)

def remove_same_range_features(X, x):
    col = x.name
    feature_names = [f for f in re.split(r'_(x|/|\+|\-)_', col) if f not in {'*', '/', '+', '-'}]

    return X[feature_names].corrwith(x, method='spearman').max()
# X_int.apply(lambda x: self.remove_same_range_features(X_num, x)).abs()

class ArithmeticPreprocessor(BasePreprocessor):
    def __init__(
        self,
        # Base parameters
        max_order: int = 3,
        cat_as_num: bool = False,
        min_cardinality: int = 3,
        max_base_feats: int = 150,  # TODO: Need to implement a better heuristic than choosing randomly
        max_new_feats: int = 2000,
        random_state: int = 42,
        selection_method: Literal['spearman', 'random'] = 'random',

        # Efficiency parameters
        subsample: int = 100000,  # TODO: Need to implement
        reduce_memory_usage: bool = True,
        rescale_avoid_overflow: bool = True,

        # Filtering parameters
        corr_threshold: float = 0.95,
        use_cross_corr: bool = False,
        use_target_corr: bool = False,
        cross_corr_n_block_size: int = 5000,
        max_accept_for_pairwise: int = 10000,

        # Additional variation
        # scale_X: bool = False,
        # add_unary: bool = False,
        **kwargs
        ):
        super().__init__(keep_original=True)
        self.max_order = max_order
        self.cat_as_num = cat_as_num
        self.min_cardinality = min_cardinality
        self.max_base_feats = max_base_feats
        self.max_new_feats = max_new_feats
        self.selection_method = selection_method
        self.subsample = subsample
        self.reduce_memory_usage = reduce_memory_usage
        self.rescale_avoid_overflow = rescale_avoid_overflow
        self.corr_threshold = corr_threshold
        self.use_cross_corr = use_cross_corr
        self.use_target_corr = use_target_corr
        self.cross_corr_n_block_size = cross_corr_n_block_size
        self.max_accept_for_pairwise = max_accept_for_pairwise

        self.rng = np.random.default_rng(random_state)

        self.timelog = TimerLog()
        self.new_feats = []

    def spearman_selection(self, X, y):
        # FIXME: Currently heavily relies on polars for performance, make sure a numpy and a polars version exists for each function
        from tabprep.preprocessors.arithmetic.filtering_polars import filter_spearman_pl_from_pandas, get_target_corr, filter_cross_correlation
        ### Apply advanced filtering steps (spearman correlation thresholding)
        # TODO: Might skip that and instead add the corr based filter to basic + use a max_base_features parameter
        with self.timelog.block("advanced_filter_base"):
            use_cols = filter_spearman_pl_from_pandas(X, corr_threshold=self.corr_threshold, verbose=True)
        print(f"Using {len(use_cols)}/{X.shape[1]} features after advanced filtering")
        X = X[use_cols]

        if X.shape[1] == 0:
            print("No features left after filtering. Exiting.")
            return self

        if self.use_target_corr:
            with self.timelog.block("target_corr_base"):
                target_corr = get_target_corr(X, y, top_k=5, use_polars=False)
            print("Top base feature correlations:")
            print(target_corr)

        for order in range(2, self.max_order+1):
            if order > X.shape[1]:
                break
            print('---' * 20)
            print(f"Generating order {order} interaction features")

            # 6. Generate higher-order interaction features
            with self.timelog.block(f"get_interactions_{order}-order"):
                if order == 2:
                    X_int_higher = get_all_bivariate_interactions(X, order=2, max_base_interactions=int(self.max_accept_for_pairwise / 5))
                    X_int = X.copy()
                else:
                    X_int_higher = add_higher_interaction(X, X_int, max_base_interactions=int(self.max_accept_for_pairwise / 5))

            with self.timelog.block(f"reduce_memory_{order}-order"):
                X_int_higher = reduce_memory_usage(X_int_higher, verbose=True, rescale=False)
            print(f"Generated {X_int_higher.shape[1]} {order}-order interaction features")

            if self.use_target_corr:
                with self.timelog.block(f"target_corr_{order}-order"):
                    target_corr = get_target_corr(X_int_higher, y, top_k=5, use_polars=False)
                print(f"Top {order}-order interaction feature correlations:")
                print(target_corr)

            # 7. Filter higher-order interaction features
            n_feats_start = X_int_higher.shape[1]
            # basic
            with self.timelog.block(f"basic_filter_{order}-order"):
                X_int_higher = basic_filter(X_int_higher, use_polars=False, min_cardinality=self.min_cardinality)
            print(f"Using {len(X_int_higher.columns)}/{n_feats_start} features after basic filtering")

            # based on correlations among interaction features
            if X_int_higher.shape[1] > self.max_accept_for_pairwise:
                print(f"Limiting interaction features to {self.max_accept_for_pairwise} (from {X_int_higher.shape[1]})")
                X_int_higher = X_int_higher.sample(n=self.max_accept_for_pairwise, random_state=42, axis=1)

            n_feats_start = X_int_higher.shape[1]
            with self.timelog.block(f"spearman_int_filter_{order}-order"):
                use_cols = filter_spearman_pl_from_pandas(X_int_higher, corr_threshold=self.corr_threshold, verbose=True)
            X_int_higher = X_int_higher[use_cols]
            print(f"Using {len(use_cols)}/{n_feats_start} features after interaction feature correlation filtering")

            # based on cross-correlation with base features
            if self.use_cross_corr:
                n_feats_start = X_int_higher.shape[1]
                with self.timelog.block(f"cross_correlation_{order}-order"):
                    use_cols = filter_cross_correlation(X_int, X_int_higher, corr_threshold=self.corr_threshold, block_size=self.cross_corr_n_block_size)
                X_int_higher = X_int_higher[use_cols]
                print(f"Using {len(use_cols)}/{n_feats_start} features after cross-correlation filtering")

            if self.use_target_corr:
                target_corr = get_target_corr(X_int_higher, y, top_k=5, use_polars=False)
                print(f"Top {order}-order interaction feature correlations after filtering:")
                print(target_corr)

            X_int = X_int_higher # pd.concat([X_int, X_int_higher], axis=1)

            self.new_feats.extend(use_cols)

            if len(self.new_feats) >= self.max_new_feats:
                print(f"Reached max new features limit of {self.max_new_feats}. Stopping.")
                break

    def random_selection(self, X, y):
        for order in range(2, self.max_order+1):
            if order > X.shape[1]:
                break
            print('---' * 20)
            print(f"Generating order {order} interaction features")

            # 6. Generate higher-order interaction features
            with self.timelog.block(f"get_interactions_{order}-order"):
                if order == 2:
                    X_int_higher = get_all_bivariate_interactions(X, order=2, max_base_interactions=int(self.max_new_feats/5))
                    X_int = X.copy()
                else:
                    X_int_higher = add_higher_interaction(X, X_int, max_base_interactions=int(self.max_new_feats/5))

            with self.timelog.block(f"reduce_memory_{order}-order"):
                X_int_higher = reduce_memory_usage(X_int_higher, verbose=True, rescale=False)
            print(f"Generated {X_int_higher.shape[1]} {order}-order interaction features")

            X_int = X_int_higher # pd.concat([X_int, X_int_higher], axis=1)

            self.new_feats.extend(X_int_higher.columns.tolist())

            if len(self.new_feats) >= self.max_new_feats:
                print(f"Reached max new features limit of {self.max_new_feats}. Stopping.")
                break

    def _fit(self, X_in, y_in = None, **kwargs):
        # TODO: Add a check that the original features names don't contain arithmetic operators to avoid issues in transform
        X = X_in.copy()
        y = y_in.copy() if y_in is not None else None

        if self.subsample < X.shape[0]:
            X = X.sample(n=self.subsample, random_state=self.rng, axis=0)
            if y is not None:
                y = y.loc[X.index]

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True) if y is not None else None

        ### Reduce memory usage
        if self.reduce_memory_usage:
            with self.timelog.block("reduce_memory_usage_base"):
                X = reduce_memory_usage(X, verbose=True, rescale=self.rescale_avoid_overflow)

        ### if categorical-as-numerical is enabled, convert categorical features to numerical
        if self.cat_as_num:
            self.cat_as_num_preprocessor = CatAsNumTransformer(keep_original=False)
            X = self.cat_as_num_preprocessor.fit_transform(X)
        else: 
            X = X.select_dtypes(include=[np.number])
            if X.shape[1]==0:
                print('No numeric features available. Exiting.')
                return self

        ### Apply basic filtering steps
        n_base_feats_start = X.shape[1]
        with self.timelog.block("basic_filter_base"):
            X = basic_filter(X, use_polars=False, min_cardinality=self.min_cardinality) # TODO: Make data adaptive and use more restrictive threshold for large datasets
        print(f"Using {len(X.columns)}/{n_base_feats_start} features after basic filtering")
        if X.shape[1] > self.max_base_feats:
            print(f"Limiting base features to {self.max_base_feats} (from {X.shape[1]})")
            X = X.sample(n=self.max_base_feats, random_state=42, axis=1)
        self.used_base_cols = X.columns.tolist()

        if self.selection_method == 'random':
            self.random_selection(X, y)
        else:
            self.spearman_selection(X, y)

        # self.new_feats_compiled = [e.replace('_/_', '/').replace('_*_', '*').replace('_+_', '+').replace('_-_', '-') for e in self.new_feats]
        self._compile_expressions() # Assumes that self.new_feats is populated
        self.time_logs = self.timelog.summary()
        return self
    
    def _compile_expressions(self):
        # map every base column to a safe token (v0, v1, ...)
        base_cols = list(self.used_base_cols)  # or whatever you track
        self.col_map = {c: f'v{i}' for i, c in enumerate(base_cols)}
        self.inv_col_map = {v: k for k, v in self.col_map.items()}  # optional

        # one regex to replace exact column names (longest-first to avoid substr hits)
        pat = re.compile('|'.join(map(re.escape, sorted(self.col_map, key=len, reverse=True))))

        def _normalize(expr: str) -> str:
            expr = (expr
                    .replace('_*_', '*')
                    .replace('_/_', '/')
                    .replace('_+_', '+')
                    .replace('_-_', '-'))
            return pat.sub(lambda m: self.col_map[m.group(0)], expr)

        self.new_feats_compiled = {name: _normalize(name) for name in self.new_feats}
        
    def _transform(self, X_in, **kwargs):
        X = X_in.copy()
        if len(self.new_feats) == 0:
            return pd.DataFrame(index=X.index)

        X = X[self.used_base_cols]
        if self.cat_as_num:
            X = self.cat_as_num_preprocessor.transform(X)

        local = {tok: X[orig].to_numpy(dtype='float64', copy=False)
                for orig, tok in self.col_map.items()}
        new = {name: pd.eval(expr, local_dict=local, engine="numexpr")
            for name, expr in self.new_feats_compiled.items()}
        X_out = pd.DataFrame(new, index=X.index)
        return X_out.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":

    import openml
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper
    from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
    from tabprep.utils.modeling_utils import adjust_target_format

    metadata = load_task_metadata()
    dataset_names = metadata.sort_values('n_samples_train_per_fold').name.tolist()

# ['blood-transfusion-service-center', 'diabetes', 'anneal', 'QSAR_fish_toxicity', 'credit-g', 'maternal_health_risk', 'concrete_compressive_strength', 
# 'qsar-biodeg', 'healthcare_insurance_expenses', 'website_phishing', 'Fitness_Club', 'airfoil_self_noise', 'Another-Dataset-on-used-Fiat-500', 'MIC', 
# 'Is-this-a-good-customer', 'Marketing_Campaign', 'hazelnut-spread-contaminant-detection', 'seismic-bumps', 'splice', 'Bioresponse', 'hiva_agnostic', 
# 'students_dropout_and_academic_success', 'churn', 'QSAR-TID-11', 'polish_companies_bankruptcy', 'wine_quality', 'taiwanese_bankruptcy_prediction', 
# 'NATICUSdroid', 'coil2000_insurance_policies', 'Bank_Customer_Churn', 'heloc', 'jm1', 'E-CommereShippingData', 'online_shoppers_intention',
#  'in_vehicle_coupon_recommendation', 'miami_housing', 'HR_Analytics_Job_Change_of_Data_Scientists', 'houses', 'superconductivity', 
# 'credit_card_clients_default', 'Amazon_employee_access', 'bank-marketing', 'Food_Delivery_Time', 'physiochemical_protein', 'kddcup09_appetency',
#  'diamonds', 'Diabetes130US', 'APSFailure', 'SDSS17', 'customer_satisfaction_in_airline', 'GiveMeSomeCredit']
    # dataset_names = ['concrete_compressive_strength']
    dataset_name = 'diabetes' 
    for num, dataset_name in enumerate(dataset_names):
        if num < 45:
            continue
        print('==='*20)
        print(f"Processing {num} dataset: {dataset_name}")
        print('==='*20)

        tid = int(metadata.loc[metadata.name==dataset_name,'tid'].iloc[0])
        task = OpenMLTaskWrapper(openml.tasks.get_task(tid))

        fold = 0
        repeat = 0
        sample = None

        X, y, X_test, y_test = task.get_train_test_split(fold=fold, repeat=repeat)
        target_type = task.problem_type

        y = adjust_target_format(y, target_type)
        y_test = adjust_target_format(y_test, target_type)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        print(f"Dataset: {dataset_name}, n_samples: {X.shape[0]}, n_features: {X.shape[1]}")
        prep = ArithmeticPreprocessor()
        prep.fit(X, y)
        X_new = prep.transform(X)
