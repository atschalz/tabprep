import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tabprep.utils import sample_from_set
from itertools import combinations 
from tabprep.utils import make_cv_function, p_value_wilcoxon_greater_than_zero
from tabprep.proxy_models import TargetMeanRegressor, TargetMeanClassifier
from itertools import product, combinations
from category_encoders import LeaveOneOutEncoder
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from tabprep.base_preprocessor import BasePreprocessor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import OrdinalEncoder
from typing import Literal

from autogluon.features.generators.memory_minimize import CategoryMemoryMinimizeFeatureGenerator

class CategoricalFeatureEngineer(BasePreprocessor):
    # TODO: Write a common class for different categorical feature preprocessing modules
    # TODO: Add memory estimation and function to process columns in chunks if memory is not enough
    # TODO: Make sure 'independent' and 'expand' modes work
    # TODO: Include groupby interactions
    def __init__(self, 
                 target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='average', mvp_max_cols_use=100, random_state=42, verbose=True,
                 execution_mode: Literal['independent', 'reduce', 'expand', 'all'] = 'independent', 
                 max_order=2, num_operations='all',
                 scores: dict = None, 
                 min_cardinality=6,
                 min_frequency_new_cat=5,
                 use_mvp=True,
                 ):
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, random_state=random_state, verbose=verbose)
        self.execution_mode = execution_mode
        self.max_order = max_order
        self.num_operations = num_operations
        self.scores = scores
        self.min_cardinality = min_cardinality
        self.min_frequency_new_cat = min_frequency_new_cat
        self.use_mvp = use_mvp

        if self.target_type=='regression':
            self.target_model = TargetMeanRegressor()
        else:
            self.target_model = TargetMeanClassifier()
        
        if scores is None:
            self.scores = {}

        self.significances = {}
        self.new_col_set = []
        self.base_cat_cols = []

    def combine(self, X_in, order=2, num_operations='all', seed=42, **kwargs):
        # TODO: Implement as matrix operations to speed up the process
        self.cnt_map = {}
        X = X_in.copy()
        X = X.astype('U')
        feat_combs = set(combinations(np.unique(X.columns), order))

        if num_operations == "all":
            feat_combs_use = feat_combs
        else:
            feat_combs_use = sample_from_set(feat_combs, num_operations)
        feat_combs_use_arr = np.array(list(feat_combs_use)).transpose()

        new_cols = []
        for col1, col2 in zip(feat_combs_use_arr[0], feat_combs_use_arr[1]):
            if col1 == col2:
                continue
            curr_cols = [x.name for x in new_cols]
            new_col = col2 + "_&_" + col1
            if new_col in curr_cols:
                continue
            if col1 + "_&_" + col2 in curr_cols:
                continue
            self.cnt_map[col1 + "_by_" + col2] = X[col1].groupby(X[col2]).count()
            one_by_two = X[col2].map(self.cnt_map[col1 + "_by_" + col2])
            one_by_two.name = col1 + "_by_" + col2
            new_cols.append(one_by_two)

            self.cnt_map[col2 + "_by_" + col1] = X[col2].groupby(X[col1]).count()
            two_by_one = X[col1].map(self.cnt_map[col2 + "_by_" + col1])
            two_by_one.name = col2 + "_by_" + col1
            new_cols.append(two_by_one)

            one_and_two = X[col1]+ "_&_"+X[col2]
            one_and_two.name = col1 + "_&_" + col2
            new_cols.append(one_and_two)

        X = pd.concat([X] + new_cols, axis=1)
        return X
    
    def groupby(self, X_in, order=2, num_operations='all', seed=42, **kwargs):
        # TODO: Implement as matrix operations to speed up the process
        self.cnt_map = {}
        X = X_in.copy()
        X = X.astype('U')
        feat_combs = set(combinations(np.unique(X.columns), order))

        if num_operations == "all":
            feat_combs_use = feat_combs
        else:
            feat_combs_use = sample_from_set(feat_combs, num_operations)
        feat_combs_use_arr = np.array(list(feat_combs_use)).transpose()

        new_cols = []
        for col1, col2 in zip(feat_combs_use_arr[0], feat_combs_use_arr[1]):
            if col1 == col2:
                continue
            curr_cols = [x.name for x in new_cols]
            new_col = col2 + "_&_" + col1
            if new_col in curr_cols:
                continue
            if col1 + "_&_" + col2 in curr_cols:
                continue
            self.cnt_map[col1 + "_by_" + col2] = X[col1].groupby(X[col2]).count()
            one_by_two = X[col2].map(self.cnt_map[col1 + "_by_" + col2])
            one_by_two.name = col1 + "_by_" + col2
            new_cols.append(one_by_two)

            self.cnt_map[col2 + "_by_" + col1] = X[col2].groupby(X[col1]).count()
            two_by_one = X[col1].map(self.cnt_map[col2 + "_by_" + col1])
            two_by_one.name = col2 + "_by_" + col1
            new_cols.append(two_by_one)

            one_and_two = X[col1]+ "_&_"+X[col2]
            one_and_two.name = col1 + "_&_" + col2
            new_cols.append(one_and_two)

        X = pd.concat([X] + new_cols, axis=1)

        return X

    # def combine_predefined(self, X_in, comb_lst, **kwargs):
    #     X = X_in.copy()
    #     X = X.astype('U')
    #     feat_combs_use = [i.split("_by_") if "by" in i else i.split("_&_")  for i in comb_lst]
    #     feat_combs_use_arr = np.unique(feat_combs_use,axis=0).transpose()

    #     new_cols = []
    #     for col1, col2 in zip(feat_combs_use_arr[0], feat_combs_use_arr[1]):
    #         if col1 == col2:
    #             continue
    #         curr_cols = [x.name for x in new_cols]
    #         new_col = col2 + "_&_" + col1
    #         if new_col in curr_cols:
    #             continue
    #         if col1 + "_&_" + col2 in curr_cols:
    #             continue
    #         self.cnt_map[col1 + "_by_" + col2] = X[col1].groupby(X[col2]).count()
    #         one_by_two = X[col2].map(self.cnt_map[col1 + "_by_" + col2])
    #         one_by_two.name = col1 + "_by_" + col2
    #         new_cols.append(one_by_two)

    #         self.cnt_map[col2 + "_by_" + col1] = X[col2].groupby(X[col1]).count()
    #         two_by_one = X[col1].map(self.cnt_map[col2 + "_by_" + col1])
    #         two_by_one.name = col2 + "_by_" + col1
    #         new_cols.append(two_by_one)

    #         one_and_two = X[col1]+ "_&_"+X[col2]
    #         one_and_two.name = col1 + "_&_" + col2
    #         new_cols.append(one_and_two)

    #     X = pd.concat([X] + new_cols, axis=1)
    #     return X

    def combine_predefined(self, X_in, comb_lst, **kwargs):
        X = X_in.copy()
        X_str = X.astype('U')
        feat_combs_use = [i.split("_&_") for i in comb_lst if "&" in i]
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

        new_cols = []
        for col1, col2 in zip(feat_combs_use_arr[0], feat_combs_use_arr[1]):
            one_and_two = X_str[col1]+ "_&_"+X_str[col2]
            one_and_two.name = col1 + "_&_" + col2
            new_cols.append(one_and_two)

        X = pd.concat([X] + new_cols, axis=1)
        return X
    
    def groupby_predefined(self, X_in, comb_lst, **kwargs):
        X = X_in.copy()
        X_str = X.astype('U')
        feat_combs_use = [i.split("_by_") for i in comb_lst if "by" in i]
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

        new_cols = []
        for col1, col2 in zip(feat_combs_use_arr[0], feat_combs_use_arr[1]):
            one_by_two = X_str[col2].map(self.cnt_map[col1 + "_by_" + col2]).fillna(0)
            one_by_two.name = col1 + "_by_" + col2
            new_cols.append(one_by_two)

        X = pd.concat([X] + new_cols, axis=1)
        return X

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

    def filter_combinations(self, X_in, **kwargs):
        X = X_in.copy()
        cols = [i for i in X.columns if '_&_'in i]

        drop_cols = []
        for col in cols:
            if X[col].value_counts().max()< self.min_frequency_new_cat:
                drop_cols.append(col)

        return X.drop(drop_cols, axis=1)
    
    def get_transform_feature(self, X_in, name, **kwargs):
        X = X_in.copy()
        # X = X.astype('U')
        if "_&_" in name:
            col1,col2 = name.split("_&_")
            new_col = X[col1]+ "_&_"+X[col2]
            new_col.name = col1 + "_&_" + col2
        elif "_by_" in name:
            col1,col2 = name.split("_by_")
            new_col = X[col2].map(self.cnt_map[col1 + "_by_" + col2]).fillna(0)
            new_col.name = col1 + "_by_" + col2
        
        return new_col
    
    def fit(self, X_in, y_in=None):
        # TODO: Add logic to filter correlated new features early on
        # TODO: Add three different execution modes: sequential independent, sequential with performant subset, and sequential with lower order as new features
        # TODO: Add leave-one-out test to get rid of trivial features
        # TODO: Add test to filter new cat features with few unique values

        
        X = X_in.copy()
        y = y_in.copy()

        y = self.adjust_target_format(y)

        X = X.select_dtypes(include=['category', 'object'])#.astype('U') #TODO: Check whether astype assignment at this point is best

        print(f"Removing {X.shape[1] - X.loc[:, X.nunique() > self.min_cardinality].shape[1]} low cardinality features with less than {self.min_cardinality} unique values")
        X = X.loc[:, X.nunique() > self.min_cardinality]  # Remove low cardinality features

        if X.shape[1] < 2:
            print("Less than 2 categorical features found. Returning original DataFrame.")
            self.new_col_set = []
            self.single_cnt_maps = {}
            return self

        self.base_cat_cols = X.columns.tolist()

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
        elif self.execution_mode == "all":
            X_new = self.combine(X, order=np.min([2, self.max_order]), num_operations=self.num_operations)
            # X_new = pd.concat([X, X_new], axis=1)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}. Use 'sequential' or 'independent'.")
        
        X_new = self.filter_combinations(X_new)

        self.new_col_set = list(set(X_new.columns)-set(X.columns))
        
        if self.use_mvp:
            self.new_col_set = self.multivariate_performance_test(X_new, y, test_cols=self.new_col_set, max_cols_use=self.mvp_max_cols_use, individual_test_condition='never')

        cat_cols = X_new.select_dtypes(include=['category', 'object']).columns
        
        self.single_cnt_maps = {}
        self.ordinal = {}
        self.dtype_map = {}
        for col in cat_cols:
            self.single_cnt_maps[col] = X_new[col].astype('U').value_counts()
            self.ordinal[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_new[[col]])
            X_new[col] = self.ordinal[col].transform(X_new[[col]])
            self.dtype_map[col] = X_new[col].astype('category').dtype

        return self

    def transform(self, X_in):        
        X_out = X_in.copy()

        if len(self.new_col_set) > 0:
            comb_cols = [col for col in self.new_col_set if "_&_" in col]
            group_cols = [col for col in self.new_col_set if "_by_" in col]
            if len(comb_cols) > 0:
                X_out = self.combine_predefined(X_out, comb_cols)
            if len(group_cols) > 0:
                X_out = self.groupby_predefined(X_out, group_cols)

            new_cat_cols = X_out[self.new_col_set].select_dtypes(include=['category', 'object']).columns.tolist()
            new_cols = []
            for col in self.base_cat_cols+new_cat_cols:
                if col in self.single_cnt_maps:
                    new_col = X_out[col].astype('U').map(self.single_cnt_maps[col]).fillna(0)
                    new_col.name = col + "_cnt"
                    new_cols.append(new_col)            
                if col in new_cat_cols:
                    X_out[col] = self.ordinal[col].transform(X_out[[col]])
                    X_out[col] = X_out[col].astype(self.dtype_map[col])
                
            X_out = pd.concat([X_out] + new_cols, axis=1)

        return X_out[np.unique(X_out.columns)]


if __name__ == "__main__":
    import os
    from tabprep.utils import *
    import openml
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'Marketing'
    for benchmark in ['TabArena']: # ["Grinsztajn", "TabArena", "TabZilla"]:
        exp_name = f"EXP_catengineer{benchmark}"
        if False: #os.path.exists(f"{exp_name}.pkl"):
            with open(f"{exp_name}.pkl", "rb") as f:
                results = pickle.load(f)
        else:
            results = {}
            results['performance'] = {}
            results['new_cols'] = {}
            results['significances'] = {}

        tids, dids = get_benchmark_dataIDs(benchmark)  

        remaining_cols = {}

        for tid, did in zip(tids, dids):
            task = openml.tasks.get_task(tid)  # to check if the datasets are available
            data = openml.datasets.get_dataset(did)  # to check if the datasets are available
            # if dataset_name not in data.name:
            #     continue
        
            
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
            
            # X = X.sample(n=1000)
            # y = y.loc[X.index]

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
            
            detector = CategoricalFeatureEngineer(        
                target_type=target_type,
                execution_mode='all',
                max_order=3,
                use_mvp=True,
            )

            detector.fit(X, y)
            X_new = detector.transform(X)
            print(f"New columns ({X_new.shape[1] - X.shape[1]}): {list(set(X_new.columns)-set(X.columns))}")
            # print(pd.DataFrame(detector.significances))
            # print(pd.DataFrame({col: pd.DataFrame(detector.scores[col]).mean().sort_values() for col in detector.scores}).transpose())
            # print(detector.linear_features.keys())
            results['performance'][data.name] = detector.scores
            results['significances'][data.name] = detector.significances
            results['new_cols'][data.name] = list(set(X_new.columns)-set(X.columns))
            
        with open(f"{exp_name}.pkl", "wb") as f:
            pickle.dump(results, f)
        break


