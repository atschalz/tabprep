from tabprep.preprocessors.arithmetic.combinations import *
from tabprep.preprocessors.arithmetic.filtering import *
from tabprep.preprocessors.arithmetic.memory import *
from tabprep.preprocessors.base import BasePreprocessor
import re
from pandas.api.types import is_numeric_dtype

from time import perf_counter
from contextlib import contextmanager

from numba import njit, prange

from math import comb

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

    def summary(self, verbose=False):
        if verbose:
            print("\n--- Timing Summary (in order) ---")
            for name, total in self.times.items():
                print(f"{name:<20} {total:.3f}s")
        return dict(self.times)

def remove_same_range_features(X, x):
    col = x.name
    feature_names = [f for f in re.split(r'_(\*|/|\+|\-)_', col) if f not in {'*', '/', '+', '-'}]

    return X[feature_names].corrwith(x, method='spearman').max()

# Map from name-encoded tokens to actual ops
OP_TOKENS = {
    '_+_': '+',
    '_-_': '-',
    '_*_': '*',
    '_/_': '/',
}

# Compact op codes for numba
OP_CODE = {
    '+': 0,
    '-': 1,
    '*': 2,
    '/': 3,
}


def parse_feature_expr(name: str, base_idx: dict):
    """
    Parse a feature name like 'colA_*_colB_*_colC' into:
      - indices: list[int] of base column indices
      - ops:     list[int] of op codes between them (len = order-1)
    Returns (indices, op_codes) or (None, None) if unparsable.
    """
    expr = name
    for tok in OP_TOKENS.keys():
        expr = expr.replace(tok, f' {tok} ')

    parts = expr.split()
    if not parts:
        return None, None

    operands = parts[0::2]   # col names
    op_tokens = parts[1::2]  # '_+_', '_*_', ...

    if len(op_tokens) != max(0, len(operands) - 1):
        return None, None

    try:
        indices = [base_idx[col] for col in operands]
    except KeyError:
        return None, None

    ops = []
    for tok in op_tokens:
        op_char = OP_TOKENS.get(tok)
        if op_char is None or op_char not in OP_CODE:
            return None, None
        ops.append(OP_CODE[op_char])

    return indices, ops


@njit(parallel=True, fastmath=True)
def eval_order_fused(X_base_T, idx_mat, op_mat):
    """
    Fused, parallel evaluation for all features of a given order.

    X_base_T: (n_base, n_rows) float64, transposed base matrix
    idx_mat:  (n_feats, order) int32, column indices into X_base_T
    op_mat:   (n_feats, order-1) int8, op codes between operands for each feature

    Returns: (n_rows, n_feats) float64
    """
    n_base, n_rows = X_base_T.shape
    n_feats, order = idx_mat.shape

    out = np.empty((n_rows, n_feats), dtype=X_base_T.dtype)

    for i in prange(n_rows):
        for f in range(n_feats):
            # First operand
            idx0 = idx_mat[f, 0]
            val = X_base_T[idx0, i]

            # Fold remaining operands
            for k in range(1, order):
                idxk = idx_mat[f, k]
                b = X_base_T[idxk, i]
                op = op_mat[f, k - 1]

                if op == 0:      # +
                    val = val + b
                elif op == 1:    # -
                    val = val - b
                elif op == 2:    # *
                    val = val * b
                else:            # /
                    if b == 0.0:
                        val = np.nan
                    else:
                        val = val / b

            out[i, f] = val

    return out

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
        interaction_types: list[str] = ['/', '*', '-', '+'],
        remove_constant_mostlynan: bool = True,

        # Efficiency parameters
        subsample: int = 100000,  # TODO: Need to implement
        reduce_memory: bool = True,
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
        verbose: bool = False,
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
        self.reduce_memory = reduce_memory
        self.rescale_avoid_overflow = rescale_avoid_overflow
        self.corr_threshold = corr_threshold
        self.use_cross_corr = use_cross_corr
        self.use_target_corr = use_target_corr
        self.cross_corr_n_block_size = cross_corr_n_block_size
        self.max_accept_for_pairwise = max_accept_for_pairwise
        self.verbose = verbose
        self.interaction_types = interaction_types
        self.remove_constant_mostlynan = remove_constant_mostlynan

        for i in self.interaction_types:
            if i not in OP_CODE:
                raise ValueError(f"Unsupported interaction type: {i}")

        self.rng = np.random.default_rng(random_state)

        self.timelog = TimerLog()
        self.new_feats = []
        self.order_batches = {}  # order -> {'idx': np.ndarray, 'ops': np.ndarray, 'names': list[str]}

    def estimate_no_of_new_features(self, X: pd.DataFrame) -> int:
        # 1. Determine the no. of base features
        pass_cardinality_filter = X.nunique().values>=self.min_cardinality
        pass_cat_filter = X.apply(is_numeric_dtype).values if not self.cat_as_num else np.array([True]*X.shape[1])
        base_feat_mask = pass_cardinality_filter & pass_cat_filter
        num_base_feats = min(np.sum(base_feat_mask), self.max_base_feats)

        # 2. Estimate the no. of new arithmetic features per order
        no_interaction_types = len(self.interaction_types)
        num_new_feats = 0
        for order in range(2, self.max_order+1):
            if order > num_base_feats:
                break
            if order == 2: 
                if '/' in self.interaction_types:
                    no_interaction_types += 1
                num_new_feats = comb(num_base_feats, 2)*no_interaction_types #num_base_feats*(num_base_feats-1)/2*no_interaction_types
                if '/' in self.interaction_types:
                    no_interaction_types -= 1
            else:
                num_new_feats += ((((num_base_feats-2)*(num_new_feats))) * no_interaction_types)
            if num_new_feats > self.max_new_feats:
                num_new_feats = self.max_new_feats
                break
        return int(num_new_feats)

    def spearman_selection(self, X, y):
        # FIXME: Currently heavily relies on polars for performance, make sure a numpy and a polars version exists for each function
        from tabprep.preprocessors.arithmetic.filtering_polars import filter_spearman_pl_from_pandas, get_target_corr, filter_cross_correlation
        ### Apply advanced filtering steps (spearman correlation thresholding)
        # TODO: Might skip that and instead add the corr based filter to basic + use a max_base_features parameter
        with self.timelog.block("advanced_filter_base"):
            use_cols = filter_spearman_pl_from_pandas(X, corr_threshold=self.corr_threshold, verbose=self.verbose)
            
        if self.verbose:
            print(f"Using {len(use_cols)}/{X.shape[1]} features after advanced filtering")
        X = X[use_cols]

        if X.shape[1] == 0:
            if self.verbose:
                print("No features left after filtering. Exiting.")
            return self

        if self.use_target_corr:
            with self.timelog.block("target_corr_base"):
                target_corr = get_target_corr(X, y, top_k=5, use_polars=False)
            if self.verbose:
                print("Top base feature correlations:")
                print(target_corr)

        for order in range(2, self.max_order+1):
            if order > X.shape[1]:
                break
            if self.verbose:
                print('---' * 20)
                print(f"Generating order {order} interaction features")

            # 6. Generate higher-order interaction features
            with self.timelog.block(f"get_interactions_{order}-order"):
                if order == 2:
                    X_int_higher = get_all_bivariate_interactions(X, order=2, max_base_interactions=int(self.max_accept_for_pairwise / 5), random_state=self.rng, interaction_types=self.interaction_types)
                    X_int = X.copy()
                else:
                    X_int_higher = add_higher_interaction(X, X_int, max_base_interactions=int(self.max_accept_for_pairwise / 5), random_state=self.rng, interaction_types=self.interaction_types)

            if self.reduce_memory:
                with self.timelog.block(f"reduce_memory_{order}-order"):
                    X_int_higher = reduce_memory_usage(X_int_higher, rescale=False, verbose=self.verbose)
            if self.verbose:
                print(f"Generated {X_int_higher.shape[1]} {order}-order interaction features")

            if self.use_target_corr:
                with self.timelog.block(f"target_corr_{order}-order"):
                    target_corr = get_target_corr(X_int_higher, y, top_k=5, use_polars=False)
                if self.verbose:
                    print(f"Top {order}-order interaction feature correlations:")
                    print(target_corr)

            # 7. Filter higher-order interaction features
            n_feats_start = X_int_higher.shape[1]
            # basic
            with self.timelog.block(f"basic_filter_{order}-order"):
                X_int_higher = basic_filter(X_int_higher, use_polars=False, min_cardinality=self.min_cardinality, remove_constant_mostlynan=self.remove_constant_mostlynan)
            if self.verbose:
                print(f"Using {len(X_int_higher.columns)}/{n_feats_start} features after basic filtering")

            # based on correlations among interaction features
            if X_int_higher.shape[1] > self.max_accept_for_pairwise:
                if self.verbose:
                    print(f"Limiting interaction features to {self.max_accept_for_pairwise} (from {X_int_higher.shape[1]})")
                X_int_higher = X_int_higher.sample(n=self.max_accept_for_pairwise, random_state=42, axis=1)

            n_feats_start = X_int_higher.shape[1]
            with self.timelog.block(f"spearman_int_filter_{order}-order"):
                use_cols = filter_spearman_pl_from_pandas(X_int_higher, corr_threshold=self.corr_threshold, verbose=True)
            X_int_higher = X_int_higher[use_cols]
            if self.verbose:
                print(f"Using {len(use_cols)}/{n_feats_start} features after spearman filtering")

            # based on cross-correlation with base features
            if self.use_cross_corr:
                n_feats_start = X_int_higher.shape[1]
                with self.timelog.block(f"cross_correlation_{order}-order"):
                    use_cols = filter_cross_correlation(X_int, X_int_higher, corr_threshold=self.corr_threshold, block_size=self.cross_corr_n_block_size)
                X_int_higher = X_int_higher[use_cols]
                if self.verbose:
                    print(f"Using {len(use_cols)}/{n_feats_start} features after cross-correlation filtering")

            if self.use_target_corr:
                target_corr = get_target_corr(X_int_higher, y, top_k=5, use_polars=False)
                if self.verbose:
                    print(f"Top {order}-order interaction feature correlations after filtering:")
                    print(target_corr)

            X_int = X_int_higher # pd.concat([X_int, X_int_higher], axis=1)

            self.new_feats.extend(use_cols)

            if len(self.new_feats) >= self.max_new_feats:
                if self.verbose:
                    print(f"Reached max new features limit of {self.max_new_feats}. Stopping.")
                break

    def random_selection(self, X, y):
        X_dict = {1: X}
        
        for order in range(2, self.max_order+1):
            if order > X.shape[1]:
                break
            if self.verbose:
                print('---' * 20)
                print(f"Generating order {order} interaction features")

            # 6. Generate higher-order interaction features
            with self.timelog.block(f"get_interactions_{order}-order"):
                if order == 2:
                    X_dict[2] = get_all_bivariate_interactions(X, order=2, max_feats=self.max_new_feats, random_state=self.rng, interaction_types=self.interaction_types)
                    # X_dict[order] = add_higher_interaction(X, X, max_feats=self.max_new_feats, random_state=self.rng, interaction_types=self.interaction_types)
                else:
                    X_dict[order] = add_higher_interaction(X, X_dict[order-1], max_feats=self.max_new_feats - X_dict[order-1].shape[1], random_state=self.rng, interaction_types=self.interaction_types)

            with self.timelog.block(f"reduce_memory_{order}-order"):
                X_dict[order] = reduce_memory_usage(X_dict[order], rescale=False, verbose=self.verbose)
                # X_dict[order] = basic_filter(X_dict[order], use_polars=False, min_cardinality=self.min_cardinality, remove_constant_mostlynan=self.remove_constant_mostlynan)
            if self.verbose:
                print(f"Generated {X_dict[order].shape[1]} {order}-order interaction features")

            self.new_feats.extend(X_dict[order].columns.tolist())

            if len(self.new_feats) >= self.max_new_feats:
                self.new_feats = self.new_feats[:self.max_new_feats]
                if self.verbose:
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

        ### if categorical-as-numerical is enabled, convert categorical features to numerical
        if self.cat_as_num:
            self.cat_as_num_preprocessor = CatAsNumTransformer(keep_original=False)
            X = self.cat_as_num_preprocessor.fit_transform(X)
        else: 
            X = X.select_dtypes(include=[np.number])
            if X.shape[1]==0:
                if self.verbose:
                    print('No numeric features available. Exiting.')
                return self

        if X.shape[1] > self.max_base_feats:
            if self.verbose:
                print(f"Limiting base features to {self.max_base_feats} (from {X.shape[1]})")
            X = X.sample(n=self.max_base_feats, random_state=42, axis=1)

        ### Reduce memory usage
        if self.reduce_memory:
            with self.timelog.block("reduce_memory_usage_base"):
                X = reduce_memory_usage(X, rescale=self.rescale_avoid_overflow, verbose=self.verbose)

        ### Apply basic filtering steps
        n_base_feats_start = X.shape[1]
        with self.timelog.block("basic_filter_base"):
            X = basic_filter(X, use_polars=False, min_cardinality=self.min_cardinality, remove_constant_mostlynan=self.remove_constant_mostlynan) # TODO: Make data adaptive and use more restrictive threshold for large datasets
        if self.verbose:
            print(f"Using {len(X.columns)}/{n_base_feats_start} features after basic filtering")
        self.used_base_cols = X.columns.tolist()

        if self.selection_method == 'random':
            self.random_selection(X, y)
        else:
            self.spearman_selection(X, y)

        self._prepare_order_batches()
        self._warmup_fused_orders()

        self.time_logs = self.timelog.summary(verbose=self.verbose)

        return self
    
    def _post_hoc_adjust_new_feats(self, new_feats):
        # In case some new features were removed after fit (e.g., due to memory issues), adjust the new_feats list
        self.new_feats = [f for f in self.new_feats if f in new_feats]
        self._prepare_order_batches()
        self._warmup_fused_orders()

    def _warmup_fused_orders(self):
        """
        Trigger Numba JIT for each order group during fit(),
        so transform() is consistently fast.
        """
        if not self.order_batches:
            return

        # tiny dummy data: 2 rows, same number of base cols
        dummy_X_T = np.zeros((len(self.used_base_cols), 2), dtype=np.float64)

        for order, batch in self.order_batches.items():
            idx_mat = batch['idx']
            ops_mat = batch['ops']
            if idx_mat.size == 0:
                continue
            eval_order_fused(dummy_X_T, idx_mat, ops_mat)


    def _prepare_order_batches(self):
        """
        Build per-order fused execution plans from self.new_feats.

        For each interaction feature, we parse:
        - its base column indices
        - the sequence of op codes between operands

        and group them by interaction order.
        """
        self.order_batches = {}

        base_idx = {col: i for i, col in enumerate(self.used_base_cols)}

        for name in self.new_feats:
            indices, ops = parse_feature_expr(name, base_idx)
            if indices is None:
                if self.verbose:
                    print(f"[ArithmeticPreprocessor] Skipping unparsable feature name: {name}")
                continue

            order = len(indices)
            if order < 2:
                # Interactions are typically order>=2; skip or handle separately if needed
                continue

            batch = self.order_batches.setdefault(order, {
                'idx': [],
                'ops': [],
                'names': [],
            })
            batch['idx'].append(indices)
            batch['ops'].append(ops)
            batch['names'].append(name)

        # Convert lists to numpy arrays for numba
        for order, batch in self.order_batches.items():
            idx_mat = np.asarray(batch['idx'], dtype=np.int32)
            ops_mat = np.zeros((idx_mat.shape[0], order - 1), dtype=np.int8)
            for i, ops in enumerate(batch['ops']):
                if len(ops) != order - 1:
                    # shouldn't happen if parsing is consistent
                    raise ValueError(f"Feature with order {order} has wrong ops length: {len(ops)}")
                for k, op_code in enumerate(ops):
                    ops_mat[i, k] = op_code

            batch['idx'] = idx_mat
            batch['ops'] = ops_mat

    def _warmup_fused_batches(self):
        if not self.batches:
            return
        dummy_X = np.zeros((2, len(self.used_base_cols)), dtype=np.float64)
        for (op, order), group in self.batches.items():
            idx_mat = group['indices']
            op_code = group['op_code']
            if idx_mat.size == 0:
                continue
            eval_batch_fused(dummy_X, idx_mat, op_code)

    def _transform(self, X_in, **kwargs):
        X = X_in.copy()

        if not self.order_batches:
            return pd.DataFrame(index=X.index)

        # Same preprocessing as in fit
        if self.cat_as_num:
            X = self.cat_as_num_preprocessor.transform(X)
        X = X[self.used_base_cols]

        # Base matrix and its transpose for better locality
        X_base = X.to_numpy(dtype='float64', copy=False)
        X_base_T = X_base.T   # shape: (n_base, n_rows)

        n_rows = X_base.shape[0]

        blocks = []
        col_names = []

        # Evaluate one fused batch per order
        for order in sorted(self.order_batches.keys()):
            batch = self.order_batches[order]
            idx_mat = batch['idx']   # (n_feats_order, order)
            ops_mat = batch['ops']   # (n_feats_order, order-1)

            if idx_mat.size == 0:
                continue

            block = eval_order_fused(X_base_T, idx_mat, ops_mat)  # (n_rows, n_feats_order)
            blocks.append(block)
            col_names.extend(batch['names'])

        if not blocks:
            return pd.DataFrame(index=X.index)

        out_mat = np.hstack(blocks)
        X_out = pd.DataFrame(out_mat, columns=col_names, index=X.index)
        X_out = X_out.replace([np.inf, -np.inf], np.nan)

        return X_out
