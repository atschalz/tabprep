import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tabprep.utils.misc import sample_from_set
from itertools import combinations 
from tabprep.utils.modeling_utils import make_cv_function
from tabprep.utils.eval_utils import p_value_wilcoxon_greater_than_zero
from tabprep.proxy_models import TargetMeanRegressor, TargetMeanClassifier
from itertools import product, combinations
from category_encoders import LeaveOneOutEncoder
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from tabprep.detectors.base_preprocessor import BasePreprocessor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import OrdinalEncoder

from autogluon.features.generators.memory_minimize import CategoryMemoryMinimizeFeatureGenerator

def map_with_nearest(x, mapping_dict):
    if x in mapping_dict:
        return mapping_dict[x]
    else:
        # Find the key with minimum absolute distance to x
        try:
            nearest_key = min(mapping_dict.keys(), key=lambda k: abs(k - x))
            return mapping_dict[nearest_key]
        except:
            return 0.0
        
def map_ranks_with_handling(X_cat_new, X_num_new, rank_map, col1, col2, default_rank=np.nan):
    """
    Maps ranks to new data based on precomputed rank_map, handling unseen values.
    
    Parameters:
        X_cat_new (pd.Series): New categorical feature values.
        X_num_new (pd.Series): New numerical feature values.
        rank_map (dict): Precomputed mapping from original data.
        col1 (str): Name of numerical column.
        col2 (str): Name of categorical column.
        default_rank (float): Default rank to use for unseen categories. Default is NaN.
    
    Returns:
        pd.Series: Ranks mapped to new data.
    """
    import bisect
    
    rank_key = col1 + "_by_" + col2 + "_RANK"
    cat_rank_dict = rank_map.get(rank_key, {})
    result_ranks = []

    for cat_val, num_val in zip(X_cat_new, X_num_new):
        if cat_val in cat_rank_dict:
            rank_series = cat_rank_dict[cat_val]
            # Map to closest numeric value
            sorted_vals = rank_series.index.values
            if len(sorted_vals) == 0:
                result_ranks.append(default_rank)
                continue

            # Find closest value
            idx = np.abs(sorted_vals - num_val).argmin()
            closest_val = sorted_vals[idx]
            result_ranks.append(rank_series.loc[closest_val])
        else:
            # Unseen category
            result_ranks.append(default_rank)

    return pd.Series(result_ranks, index=X_cat_new.index)


class GroupByFeatureEngineer(BasePreprocessor):
    # TODO: Write a common class for different categorical feature preprocessing modules
    # TODO: Add memory estimation and function to process columns in chunks if memory is not enough
    # TODO: Make sure 'independent' and 'expand' modes work
    # TODO: Include groupby interactions
    def __init__(self, 
                 target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='average', mvp_max_cols_use=100, random_state=42, verbose=True,
                 execution_mode='independent', # ['independent', 'reduce', 'expand', 'all]
                 max_order=2, num_operations='all',
                 scores: dict = None, 
                 min_cardinality=6,
                 min_frequency_new_cat=5,
                 use_mvp=True,
                 mean_difference=False,
                 num_as_cat=False,
                 ):
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, random_state=random_state, verbose=verbose)
        self.execution_mode = execution_mode
        self.max_order = max_order
        self.num_operations = num_operations
        self.scores = scores
        self.min_cardinality = min_cardinality
        self.min_frequency_new_cat = min_frequency_new_cat
        self.use_mvp = use_mvp
        self.mean_difference = mean_difference
        self.num_as_cat = num_as_cat
        self.__name__ = "GroupByFeatureEngineer"

        if self.target_type=='regression':
            self.target_model = TargetMeanRegressor()
        else:
            self.target_model = TargetMeanClassifier()
        
        if scores is None:
            self.scores = {}

        self.significances = {}
        self.new_col_set = []
        self.base_cat_cols = []


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
    
    def groupby(self, X_num_in, X_cat_in, seed=42, **kwargs):
        # TODO: Implement as matrix operations to speed up the process
        self.mean_map = {}
        self.std_map = {}
        self.min_map = {}
        self.max_map = {}
        self.rank_map = {}
        X_num = X_num_in.copy()
        X_cat = X_cat_in.copy()
        
        feat_combs = set(product(X_num.columns, X_cat.columns))
        feat_combs_use_arr = np.array(list(feat_combs)).transpose()

        new_cols = []
        for col1, col2 in zip(feat_combs_use_arr[0], feat_combs_use_arr[1]):
            self.mean_map[col1 + "_by_" + col2+'_MEAN'] = X_num[col1].groupby(X_cat[col2], observed=False).mean()
            new_col = X_cat[col2].map(self.mean_map[col1 + "_by_" + col2+'_MEAN']).astype(float).fillna(0)
            if self.mean_difference:
                new_col = new_col - X_num[col1]
            new_col.name = col1 + "_by_" + col2 + '_MEAN'
            new_cols.append(new_col)

            self.std_map[col1 + "_by_" + col2+'_STD'] = X_num[col1].groupby(X_cat[col2], observed=False).std()
            new_col = X_cat[col2].map(self.std_map[col1 + "_by_" + col2+'_STD']).astype(float).fillna(0)
            new_col.name = col1 + "_by_" + col2 + '_STD'
            new_cols.append(new_col)

            # self.min_map[col1 + "_by_" + col2+'_MIN'] = X_num[col1].groupby(X_cat[col2], observed=False).min()
            # new_col = X_cat[col2].map(self.min_map[col1 + "_by_" + col2+'_MIN']).astype(float).fillna(0)
            # new_col.name = col1 + "_by_" + col2 + '_MIN'
            # new_cols.append(new_col)

            # self.max_map[col1 + "_by_" + col2+'_MAX'] = X_num[col1].groupby(X_cat[col2], observed=False).max()
            # new_col = X_cat[col2].map(self.max_map[col1 + "_by_" + col2+'_MAX']).astype(float).fillna(0)
            # new_col.name = col1 + "_by_" + col2 + '_MAX'
            # new_cols.append(new_col)


            # train_ranks = X_num[col1].groupby(X_cat[col2], observed=False).rank(pct=True)
            # train_ranks.name = 'ranks'
            # train_ranks = pd.concat([X_num[col1], X_cat[col2], train_ranks.rename({col1: 'ranks'})],axis=1)
            # self.rank_map[col1 + "_by_" + col2+'_RANK'] = dict(train_ranks.groupby([col2,col1])['ranks'].unique().apply(lambda x: x[0] if x==x else np.nan))
            # self.rank_map[col1 + "_by_" + col2+'_RANK'] = dict(X_num[col1].groupby(X_cat[col2], observed=False).apply(lambda x: [x.rank(pct=True)]))
            # new_col = map_ranks_with_handling(X_cat[col2], X_num[col1], self.rank_map[col1 + "_by_" + col2+'_RANK'], col1, col2, default_rank=np.nan)
            # new_col.name = col1 + "_by_" + col2 + '_RANK'
            # new_cols.append(new_col)

        return pd.concat(new_cols, axis=1)
    
    def groupby_predefined(self, X_num_in, X_cat_in, 
                           col_set, mode = 'MEAN',
                           seed=42, **kwargs):
        # TODO: Implement as matrix operations to speed up the process
        X_num = X_num_in.copy()
        X_cat = X_cat_in.copy()

        if mode == 'MEAN':
            map_use = self.mean_map
        elif mode == 'STD':
            map_use = self.std_map
        elif mode == 'RANK':
            map_use = self.rank_map
        elif mode == 'MIN':
            map_use = self.min_map
        elif mode == 'MAX':
            map_use = self.max_map
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'MEAN' or 'STD'.")

        feat_combs = [i[:-len(mode)-1].split('_by_') for i in col_set]
        feat_combs_use_arr = np.array(feat_combs).transpose()

        new_cols = []
        for col1, col2 in zip(feat_combs_use_arr[0], feat_combs_use_arr[1]):
            if mode == 'RANK':
                # new_col = X_cat[col2].apply(lambda x: map_with_nearest(x, map_use)).astype(float)
                new_col = map_ranks_with_handling(X_cat, X_num, self.rank_map[col1 + "_by_" + col2+'_RANK'], col1, col2, default_rank=np.nan)

            else:
                new_col = X_cat[col2].map(map_use[col1 + "_by_" + col2 + f'_{mode}']).astype(float).fillna(0)
            if mode == 'MEAN' and self.mean_difference:
                new_col = new_col - X_num[col1]
            new_col.name = col1 + "_by_" + col2 + f'_{mode}'
            new_cols.append(new_col)

        return pd.concat(new_cols, axis=1)

    def fit(self, X_in, y_in=None):
        X = X_in.copy()
        y = y_in.copy()

        # y = self.adjust_target_format(y)

        X_cat = X.loc[:,(X.nunique()>=self.min_cardinality).values].select_dtypes(include=['category', 'object'])#.astype('U') #TODO: Check whether astype assignment at this point is best
        X_num = X.loc[:,(X.nunique()>=self.min_cardinality).values].select_dtypes(exclude=['category', 'object'])

        if len(X_cat.columns) < 1 or len(X_num.columns) < 1:
            print("Not enough categorical or numerical features found. Returning original DataFrame.")
            self.new_col_set = []
            return self

        if self.num_as_cat:
            X_cat = pd.concat([X_cat, X_num], axis=1)

        self.cat_cols = X_cat.columns.tolist()
        self.num_cols = X_num.columns.tolist()

        # print(f"Removing {X.shape[1] - X.loc[:, X.nunique() > self.min_cardinality].shape[1]} low cardinality features with less than {self.min_cardinality} unique values")
        # X = X.loc[:, X.nunique() > self.min_cardinality]  # Remove low cardinality features

        if X_cat.shape[1] < 1:
            print("No categorical features found. Returning original DataFrame.")
            return self
        if X_num.shape[1] < 1:
            print("No numerical features found. Returning original DataFrame.")
            return self

        X_new = self.groupby(X_num, X_cat, seed=self.random_state)


        # for col in X.columns:
        #     if col not in self.scores:
        #         self.scores[col] = {}
        #     if 'mean' not in self.scores[col]:
        #         self.scores[col]['mean'] = self.cv_func(X[[col]], y, Pipeline([('model', self.target_model)]))

        # if self.execution_mode == "expand":
        #     X_new = self.find_interactions_expand(X, y)
        # elif self.execution_mode == "independent":
        #     X_new = self.find_interactions_independent(X, y)
        # elif self.execution_mode == "reduce":
        #     X_new = self.find_interactions_reduce(X, y)
        # elif self.execution_mode == "all":
        #     X_new = self.combine(X, order=np.min([2, self.max_order]), num_operations=self.num_operations)
        #     # X_new = pd.concat([X, X_new], axis=1)
        # else:
        #     raise ValueError(f"Unknown execution mode: {self.execution_mode}. Use 'sequential' or 'independent'.")
        
        # X_new = self.filter_combinations(X_new)

        # self.new_col_set = list(set(X_new.columns)-set(X.columns))
        
        if self.use_mvp:
            self.new_col_set = self.multivariate_performance_test(pd.concat([X,X_new],axis=1), y, test_cols=X_new.columns, max_cols_use=self.mvp_max_cols_use, 
                                                                  individual_test_condition='never', suffix='Groupby')
        else:
            self.new_col_set = list(set(X_new.columns)-set(X.columns))

        return self

    def transform(self, X_in):        
        X_out = X_in.copy()

        if len(self.new_col_set) > 0:
            X_cat = X_out[self.cat_cols]
            X_num = X_out[self.num_cols]

            X_new_mean = self.groupby_predefined(X_num, X_cat, [i for i in self.new_col_set if '_MEAN' in i], mode='MEAN')
            # X_new_min = self.groupby_predefined(X_num, X_cat, [i for i in self.new_col_set if '_MIN' in i], mode='MIN')
            # X_new_max = self.groupby_predefined(X_num, X_cat, [i for i in self.new_col_set if '_MAX' in i], mode='MAX')
            X_new_std = self.groupby_predefined(X_num, X_cat, [i for i in self.new_col_set if '_STD' in i], mode='STD')
            # X_new = pd.concat([X_new_mean, X_new_min, X_new_max, X_new_std], axis=1)
            X_new = pd.concat([X_new_mean, X_new_std], axis=1)

            X_out = pd.concat([X_out, X_new], axis=1)

        return X_out

