from matplotlib import axis
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from fe_utils import fe_combine
from utils import sample_from_set
from itertools import combinations 
from utils import make_cv_function, p_value_wilcoxon_greater_than_zero
from proxy_models import TargetMeanRegressor, TargetMeanClassifier
from itertools import product, combinations
from category_encoders import LeaveOneOutEncoder
from sklearn.metrics import roc_auc_score, log_loss, root_mean_squared_error
from base_preprocessor import BasePreprocessor
from sklearn.dummy import DummyClassifier, DummyRegressor

class CategoricalInteractionDetector(BasePreprocessor):
    # TODO: Write a common class for different categorical feature preprocessing modules
    # TODO: Add memory estimation and function to process columns in chunks if memory is not enough
    # TODO: Make sure 'independent' and 'expand' modes work
    # TODO: Include groupby interactions
    def __init__(self, 
                 target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, random_state=42, verbose=True,
                 execution_mode='independent', # ['independent', 'reduce', 'expand']
                 max_order=2, num_operations='all',
                 scores: dict = None, 
                 min_cardinality=6,
                 ):
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, random_state=random_state, verbose=verbose)
        self.execution_mode = execution_mode
        self.max_order = max_order
        self.num_operations = num_operations
        self.scores = scores
        self.min_cardinality = min_cardinality
        
        if self.target_type=='regression':
            self.target_model = TargetMeanRegressor()
        else:
            self.target_model = TargetMeanClassifier()
        
        if scores is None:
            self.scores = {}

        self.significances = {}
        self.new_col_set = []

    def combine(self, X_in, order=2, num_operations='all', seed=42, **kwargs):
        # TODO: Implement as matrix operations to speed up the process
        X = X_in.copy()
        X = X.astype('U')
        feat_combs = set(combinations(np.unique(X.columns), order))

        if num_operations == "all":
            feat_combs_use = feat_combs
        else:
            feat_combs_use = sample_from_set(feat_combs, num_operations)
        feat_combs_use_arr = np.array(list(feat_combs_use)).transpose()

        new_names = ["_&_".join([str(i) for i in sorted(f_use)]) for f_use in feat_combs_use]

        features = X[feat_combs_use_arr[0]].values
        for num, arr in enumerate(feat_combs_use_arr[1:]):
            features += "_&_" + X[arr].values

        return pd.DataFrame(features, columns=new_names, index=X.index)
    
    def combine_predefined(self, X_in, comb_lst, **kwargs):
        X = X_in.copy()
        X = X.astype('U')
        feat_combs_use = [i.split("_&_") for i in comb_lst]
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

        features = X[feat_combs_use_arr[0]].values
        for num, arr in enumerate(feat_combs_use_arr[1:]):
            features += "_&_" + X[arr].values

        return pd.DataFrame(features, columns=comb_lst, index=X.index)

    def leave_one_out_test(self, X, y):
        if self.target_type in ['binary', 'multiclass']:
            dummy = DummyClassifier(strategy='prior')
            metric = roc_auc_score
        elif self.target_type == 'regression':
            dummy = DummyRegressor(strategy='mean')
            metric = lambda x,y: -root_mean_squared_error(x, y)

        # X_use = X.copy()
        loo_cols = []
        self.loo_scores = {}
        for cnum, col in enumerate(X.columns):                
            print(f"\rLeave-One-Out test: {cnum+1}/{len(X.columns)} columns processed", end="", flush=True)
            if cnum==0:
                dummy_pred = dummy.fit(X[[col]], y).predict(X[[col]])
                dummy_score = metric(y, dummy_pred)

            loo_pred = LeaveOneOutEncoder().fit_transform(X[col].astype('category'), y)[col]
            self.loo_scores[col] = metric(y, loo_pred)

            if self.loo_scores[col] < dummy_score:
                loo_cols.append(col)

        remaining_cols = [x for x in X.columns if x not in loo_cols]

        return remaining_cols

    def expand_combine(self, X_int_in, X_base_in, num_operations='all', seed=42, **kwargs):
        X_int = X_int_in.copy()
        X_base = X_base_in.copy()
        X_int = X_int.astype('U')
        X_base = X_base.astype('U')
        feat_combs = [(base, interact) for base, interact in product(X_base.columns, X_int.columns) if base not in interact]
        new_names = ["_&_".join(sorted(int_col.split('_&_')+[base_col])) for base_col, int_col in feat_combs]
        
        feat_combs_unique = []
        new_names_unique = []
        for f,n in zip(feat_combs, new_names):
            if n not in new_names_unique:
                feat_combs_unique.append(f)
                new_names_unique.append(n)

        if num_operations == "all":
            feat_combs_use = feat_combs_unique
        else:
            feat_combs_use = sample_from_set(feat_combs_unique, num_operations)

        # TODO: Make sure the feature values are in the same order as the columns
        feat_combs_use_arr = np.array(list(feat_combs_use)).transpose()

        features = X_base[feat_combs_use_arr[0]].values + "_&_" + X_int[feat_combs_use_arr[1]].values
        df_features = pd.DataFrame(features, columns=new_names_unique, index=X_base.index)
        df_features = df_features.T.drop_duplicates().T
        return df_features

        # for num, (base_col, int_col) in enumerate(feat_combs_use):
        #     raw_cols = int_col.split('_&_')+[base_col]
        #     name = "_&_".join(sorted(int_col.split('_&_')+[base_col]))

        #     features[name] = pd.concat([X_int[int_col], X_base[base_col]], axis=1).astype(str).apply(lambda x: "_&_".join(map(str, x)), axis=1)

        # return pd.DataFrame(features)

    def find_interactions_expand(self, X, y=None):
        # TODO: Combine the three interaction functions into one
        X_new = X.copy()

        
        for order in range(2,self.max_order+1):
            print(f"\rFind order {order} interactions", end='', flush=True)
            X_interact = self.combine(X_new, order=2, num_operations=self.num_operations)
            
            add_col_set = []
            for num, col in enumerate(X_interact.columns):
                self.significances[col] = {}
                print(f"\rProcessing column {num+1}/{len(X_interact.columns)}", end='', flush=True)
                base_cols = col.split('_&_')
                if col not in self.scores:
                    self.scores[col] = {}
                if 'mean' not in self.scores[col]:
                    self.scores[col]['mean'] = self.cv_func(X_interact[[col]], y, Pipeline([('model', self.target_model)]))

                for base_col in base_cols:
                    self.significances[col][base_col] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col]['mean'] - self.scores[base_col]['mean']
                    )

                if all(np.array(list(self.significances[col].values())) < 0.05):
                    add_col_set.append(col)
                    self.new_col_set.append(col)

            X_new = pd.concat([X_new,X_interact[add_col_set]], axis=1)
            X_new = X_new.copy() # Avoid defragmentation issues
            
            return X_new

    def find_interactions_independent(self, X, y=None):
        X_new = X.copy()
        for order in range(2,self.max_order+1):
            print(f"\rFind order {order} interactions", end='', flush=True)
            X_interact = self.combine(X, order=order, num_operations=self.num_operations)
            
            add_col_set = []
            for num, col in enumerate(X_interact.columns):
                self.significances[col] = {}
                print(f"\rProcessing column {num+1}/{len(X_interact.columns)}", end='', flush=True)
                base_cols = col.split('_&_') # Assume previous results for base cols are already computed
                
                if col not in self.scores:
                    self.scores[col] = {}
                if 'mean' not in self.scores[col]:
                    self.scores[col] = self.cv_func(X_interact[[col]], y, Pipeline([('model', self.target_model)]))

                for base_col in base_cols:
                    self.significances[col][base_col] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col]['mean'] - self.scores[base_col]['mean']
                    )

                if all(np.array(list(self.significances[col].values())) < 0.05):
                    add_col_set.append(col)
                    self.new_col_set.append(col)
            X_new = pd.concat([X_new,X_interact[add_col_set]], axis=1)
            X_new = X_new.copy() # Avoid defragmentation issues
            
            return X_new

    def find_interactions_reduce(self, X, y=None):
        X_base = X.copy()
        X_new = X.copy()
        if X.shape[1] < self.max_order:
            print(f"Warning: The number of columns ({X.shape[1]}) is less than the max order ({self.max_order}). Reducing the max order to {X_new.shape[1]}.")
            use_order = X.shape[1]
        else:
            use_order = self.max_order

        for order in range(2,use_order+1):
            print(f"\rFind order {order} interactions", end='', flush=True)
            if order == 2:
                X_interact = self.combine(X, order=2, num_operations=self.num_operations)
            else:
                X_interact = self.expand_combine(X_interact[add_col_set], X_base, num_operations=self.num_operations)
            
            loo_cols = self.leave_one_out_test(X_interact, y)
            X_interact = X_interact[loo_cols]

            add_col_set = []
            for num, col in enumerate(X_interact.columns):
                self.significances[col] = {}
                print(f"\rProcessing column {num+1}/{len(X_interact.columns)}", end='', flush=True)
                if col not in self.scores:
                    self.scores[col] = {}
                if 'mean' not in self.scores[col]:
                    self.scores[col]['mean'] = self.cv_func(X_interact[[col]], y, Pipeline([('model', self.target_model)]))

                base_cols = col.split('_&_')
                
                if order>2:
                    # TODO: Also add this logic to other modes
                    child_cols = set(combinations(base_cols, order-1))
                    base_cols += ['_&_'.join(i) for i in child_cols]
                for base_col in base_cols:
                    if not base_col in self.scores:
                        self.scores[base_col] = {}
                    if 'mean' not in self.scores[base_col]:
                        raw_cols = base_col.split('_&_')
                        self.scores[base_col]['mean'] = self.cv_func(self.combine(X[raw_cols], order=len(raw_cols)), y, Pipeline([('model', self.target_model)]))

                    self.significances[col][base_col] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col]['mean'] - self.scores[base_col]['mean']
                    )

                if all(np.array(list(self.significances[col].values())) < 0.05):
                    add_col_set.append(col)
                    self.new_col_set.append(col)
            
            if len(add_col_set) == 0:
                print(f"\rNo new interactions found for order {order}. Stopping.")
                break
            else:
                # Make sure add_col_set is unique
                add_col_set = np.unique(["_&_".join(sorted(i.split("_&_"))) for i in add_col_set]).tolist()
                X_new = pd.concat([X_new,X_interact[add_col_set]], axis=1)
                X_new = X_new.copy() # Avoid defragmentation issues

        return X_new    

    # def adapt_col_for_mvp_test(self, X_cand_in, col=None, test_cols, mode='backward'):
    #     if col is None:
    #         return X_cand_in.copy() # For cat detection, assume that all columns are given as categorical: For irrelevant cat: Use drop
    #     else:
    #         X = X_cand_in.copy()
    #         if mode == 'backward':
    #             X_out = X.drop([col], axis=1) # Drop
    #         elif mode == 'forward':
    #             X_out = X.drop(set(test_cols)-set([col]), axis=1) # Drop all test columns
    #         # For cat detection, assume that all columns are given as categorical and change to num in backward, while all to cat in forward
    #         # For irrelevant cat: Use the same drop logic
    #         return X_out

    def multivariate_performance_test(self, X_cand_in, y_in, 
                                      test_cols, max_cols_use=100):
        suffix = 'COMB'
        self.new_col_set = super().multivariate_performance_test(X_cand_in, y_in, 
                                      test_cols, suffix=suffix,  max_cols_use=max_cols_use)
                
        print(f"{len(self.new_col_set)} new columns after multivariate performance test.")
        
        rejected_cols = set(test_cols) - set(self.new_col_set)
        return X_cand_in.drop(rejected_cols,axis=1)

    def fit(self, X_in, y=None):
        # TODO: Add logic to filter correlated new features early on
        # TODO: Add three different execution modes: sequential independent, sequential with performant subset, and sequential with lower order as new features
        # TODO: Add leave-one-out test to get rid of trivial features
        X = X_in.copy()

        print(f"Removing {X.shape[1] - X.loc[:, X.nunique() > self.min_cardinality].shape[1]} low cardinality features with less than {self.min_cardinality} unique values")
        X = X.loc[:, X.nunique() > self.min_cardinality]  # Remove low cardinality features

        for col in X.columns:
            if col not in self.scores:
                self.scores[col] = {}
            if 'mean' not in self.scores[col]:
                self.scores[col]['mean'] = self.cv_func(X[[col]], y, Pipeline([('model', self.target_model)]))

        if self.execution_mode == "expand":
            X_new = self.find_interactions_expand(X, y)
        elif self.execution_mode == "independent":
            X_new = self.find_interactions_independent(X, y)
        elif self.execution_mode == "reduce":
            X_new = self.find_interactions_reduce(X, y)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}. Use 'sequential' or 'independent'.")
        
        X_new = self.multivariate_performance_test(X_new, y, test_cols=self.new_col_set, max_cols_use=self.mvp_max_cols_use)
        
        return self
    
    def transform(self, X_in):        
        X = X_in.copy()
        X = X.astype('U')

        X_out = X.copy()
        for degree in range(2, self.max_order+1):
            col_set_use = [col for col in self.new_col_set if col.count('_&_')+1 == degree]
            if len(col_set_use) > 0:
                X_degree = self.combine_predefined(X, col_set_use)
                X_out = pd.concat([X_out, X_degree], axis=1)

        return X_out
        # for num_new, col in enumerate(self.new_col_set):
        #     print(f"\rTransforming column {num_new+1}/{len(self.new_col_set)}\r", end='')
        #     base_cols = col.split('_&_')
        #     for num, base_col in enumerate(base_cols):
        #         if num == 0:
        #             X[col] = X[base_col]
        #         else:
        #             X[col] = X[col] + "_&_" + X[base_col]
        #     X = X.copy()
        # return X



if __name__ == "__main__":
    from ft_detection import clean_feature_names
    import os
    from utils import *
    import openml
    from ft_detection import clean_feature_names
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    for benchmark in ["Grinsztajn", "TabArena", "TabZilla"]:
        exp_name = f"EXP_3-order-catinteraction_{benchmark}"
        if os.path.exists(f"{exp_name}.pkl"):
            with open(f"{exp_name}.pkl", "rb") as f:
                results = pickle.load(f)
        else:
            results = {}
            results['performance'] = {}
            results['iterations'] = {}
            results['significances'] = {}

        tids, dids = get_benchmark_dataIDs(benchmark)  

        remaining_cols = {}

        for tid, did in zip(tids, dids):
            task = openml.tasks.get_task(tid)  # to check if the datasets are available
            data = openml.datasets.get_dataset(did)  # to check if the datasets are available
            if data.name in results['performance']:
                print(f"Skipping {data.name} as it already exists in results.")
                print(pd.DataFrame(results['performance'][data.name]).mean().sort_values(ascending=False))
                continue
            # else:
            #     break
            print(data.name)
            if data.name == 'guillermo':
                continue
            X, _, _, _ = data.get_data()
            y = X[data.default_target_attribute]
            X = X.drop(columns=[data.default_target_attribute])
            
            if benchmark == "Grinsztajn" and X.shape[0]>10000:
                X = X.sample(10000, random_state=0)
                y = y.loc[X.index]

            if task.task_type == "Supervised Classification":
                target_type = "binary" if y.nunique() == 2 else "multiclass"
            else:
                target_type = 'regression'
            if target_type=="multiclass":
                # TODO: Fix this hack
                y = (y==y.value_counts().index[0]).astype(int)  # make it binary
                target_type = "binary"
            elif target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
                y = (y==y.value_counts().index[0]).astype(int)  # make it numeric
            else:
                y = y.astype(float)
            
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            cat_cols = [col for col in cat_cols if X[col].nunique() > 2]
            if len(cat_cols) == 0:
                print(f"Dataset {data.name} has no candidate features, skipping...")
                continue
            
            X_cat = X.select_dtypes(include=['object', 'category'])
            X_cat = X_cat.loc[:, X_cat.nunique() >2] # remove columns with less than 3 unique values
            print(f"Dataset: {data.name} with {X_cat.shape[1]} categorical columns")
            if X_cat.shape[1] > 1:
                cd = CategoricalInteractionDetector(target_type, max_order=3, execution_mode='reduce').fit(X_cat, y)
                X_new = cd.transform(X_cat)
                # print(pd.DataFrame(cd.scores).mean().sort_values(ascending=False))
                print(f"New columns: {len(cd.new_col_set)}: {cd.new_col_set}")
            else:
                print(f"Dataset {data.name} has only one categorical column, skipping...")
                continue
            
            rem_cols = cd.new_col_set
            # rem_cols = [col for col in rem_cols if X[col].nunique() > 2]  
            print(f"{data.name} ({len(rem_cols)}): {rem_cols}")

            if len(rem_cols) == 0:
                print(f"Dataset {data.name} has no candidate features, skipping...")
                continue
            
            # if len(set(num_cols)-set(rem_cols))>100:
            #     print(f"Dataset {data.name} has too many numerical features, downsampling for efficiency.")
            #     use_cols = np.unique(np.random.choice(num_cols, size=100, replace=False).tolist()+rem_cols).tolist()
            # else:
            #     use_cols = np.unique(num_cols+rem_cols).tolist()

            # X = X[use_cols]
            # if len(rem_cols) >10:
            #     print(f"Dataset {data.name} has {len(rem_cols)} features, skipping...")
            #     continue
            # TODO: Rethink which parameters to use
            params = {
                        "objective": "binary" if target_type=="binary" else "regression",
                        "boosting_type": "gbdt",
                        "n_estimators": 1000,
                        'min_samples_leaf': 2,
                        "max_depth": 5,
                        "verbosity": -1
                    }
            cv_func = make_cv_function(target_type=target_type, verbose=False, n_folds=10)

            model = lgb.LGBMClassifier(**params) if target_type=="binary" else lgb.LGBMRegressor(**params)
            pipe = Pipeline([("model", model)])

            performances = {}
            iterations = {}
            significances = {}
            X_use = X.copy()
            X_use[cat_cols] = X_use[cat_cols].astype('category')
            all_perf, all_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
            print(f"ALL: {np.mean(all_perf):.3f}, {np.mean(all_iter):.0f} iterations")

            X_use = X.copy()
            X_use[cat_cols] = X_use[cat_cols].astype('category')
            for col in rem_cols:
                i_cols = col.split('_&_')
                for num, i_col in enumerate(i_cols):
                    if num == 0:
                        X_use[col] = X_use[i_col].astype('U')
                    else:
                        X_use[col] = (X_use[col].astype('U')+"_&_"+X_use[i_col].astype('U'))
                X_use[col] = X_use[col].astype('category')
            interact_all_perf, interact_all_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
            print(f"ALL-INTERACTIONs: {np.mean(interact_all_perf):.3f}, {np.mean(interact_all_iter):.0f} iterations")

            performances['ALL'] = all_perf
            performances['INTERACT-ALL'] = interact_all_perf
            iterations['ALL'] = all_iter
            iterations['INTERACT-ALL'] = interact_all_iter
            significances['INTERACT-ALL'] = p_value_wilcoxon_greater_than_zero(interact_all_perf-all_perf)
            if significances['INTERACT-ALL'] < 0.05:
                print(f"INTERACT-ALL is significantly better than ALL with p-value {significances['INTERACT-ALL']:.3f}")

            if len(rem_cols) > 1:
                for num, col in enumerate(rem_cols):
                    X_use = X.copy()
                    X_use[cat_cols] = X_use[cat_cols].astype('category')
                    # TODO: Make sure to prevent leaks
                    i_cols = col.split('_&_')
                    for i_num, i_col in enumerate(i_cols):
                        if i_num == 0:
                            X_use[col] = X_use[i_col].astype('U')
                        else:
                            X_use[col] = (X_use[col].astype('U')+"_&_"+X_use[i_col].astype('U'))
                    X_use[col] = X_use[col].astype('category')
                    col_perf, col_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
                    print(f"{col}: {np.mean(col_perf):.4f}, {np.mean(col_iter):.0f} iterations")

                    # performances[f'ALL-NUM'] = str(round(np.mean(num_perf), 3)) + f" ({np.std(num_perf):.3f})"
                    performances[f'{col}'] = col_perf
                    iterations[f'{col}'] = col_iter
                    print(f"Column {num+1}/{len(rem_cols)}: {col}", )
                    significances[col] = p_value_wilcoxon_greater_than_zero(col_perf-all_perf)
                    if significances[col] < 0.05:
                        print(f"{col} is significantly better than ALL-NUM with p-value {significances[col]:.3f}")

            print(pd.DataFrame(performances).mean().sort_values(ascending=False))

            results['performance'][data.name] = performances
            results['iterations'][data.name] = iterations
            results['significances'][data.name] = significances

            with open(f"{exp_name}.pkl", "wb") as f:
                pickle.dump(results, f)


