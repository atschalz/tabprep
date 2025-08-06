import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, log_loss
from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor, UnivariateLinearRegressor, UnivariateLogisticClassifier, PolynomialLogisticClassifier, PolynomialRegressor, UnivariateThresholdClassifier, MultiFeatureTargetMeanClassifier, MultiFeatureUnivariateLogisticClassifier, LightGBMBinner, KMeansBinner, TargetMeanRegressorCut, TargetMeanClassifierCut
from tabprep.utils import p_value_wilcoxon_greater_than_zero, clean_feature_names, clean_series, make_cv_function, p_value_sign_test_median_greater_than_zero, p_value_ttest_greater_than_zero, sample_from_set
import time
from category_encoders import LeaveOneOutEncoder
from tabprep.base_preprocessor import BasePreprocessor
from itertools import combinations

import pandas as pd
from scipy.stats import chi2_contingency
from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator

def cramers_v_matrix(df):
    """
    Compute a Cramér's V matrix for all categorical columns in the DataFrame.
    Optimized for performance.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    n = len(cat_cols)
    result = pd.DataFrame(np.eye(n), index=cat_cols, columns=cat_cols)  # identity matrix

    for col1, col2 in combinations(cat_cols, 2):
        table = pd.crosstab(df[col1], df[col2])
        chi2, _, _, _ = chi2_contingency(table, correction=False)
        n_obs = table.sum().sum()
        phi2 = chi2 / n_obs
        r, k = table.shape

        # Bias correction
        phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n_obs - 1))
        rcorr = r - ((r-1)**2) / (n_obs - 1)
        kcorr = k - ((k-1)**2) / (n_obs - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom == 0:
            v = np.nan
        else:
            v = np.sqrt(phi2corr / denom)

        result.loc[col1, col2] = v
        result.loc[col2, col1] = v  # fill symmetric cell

    return result



def drop_highly_correlated_features(corr_matrix: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Drops columns from the correlation matrix that are highly correlated (above threshold) with any other column,
    starting from the last column and moving backwards.

    Parameters:
    - corr_matrix (pd.DataFrame): The correlation matrix.
    - threshold (float): Correlation threshold above which columns will be dropped.

    Returns:
    - pd.DataFrame: Reduced correlation matrix with highly correlated columns dropped.
    """
    columns = corr_matrix.columns.tolist()

    i = len(columns) - 1
    while i >= 0:
        col = columns[i]
        # Check if this column is highly correlated with any of the others
        high_corr = corr_matrix[col].drop(labels=[col]).abs() > threshold
        if high_corr.any():
            # Drop the column
            corr_matrix = corr_matrix.drop(columns=col).drop(index=col)
            columns.pop(i)
        i -= 1

    return corr_matrix


class TargetRepresenter(BaseEstimator, TransformerMixin):
    def __init__(self, feature_target_rep, target_type):
        self.feature_target_rep = feature_target_rep.copy()
        self.target_type = target_type

        self.col_transformer = {}

        if self.target_type == 'binary':
            self.mean_transformer = TargetMeanClassifier
            self.linear_transformer = UnivariateLogisticClassifier
            self.mean_threshold_transformer = TargetMeanClassifierCut
        elif self.target_type == 'regression':
            self.mean_transformer = TargetMeanRegressor
            self.linear_transformer = UnivariateLinearRegressor
            self.mean_threshold_transformer = TargetMeanRegressorCut

        self.binner = LightGBMBinner
    
    def correlation_filter(self, X_in, threshold=0.95):
        corr_matrix = X_in.corr(method='spearman').abs()
        # Drop highly correlated features
        reduced_corr_matrix = drop_highly_correlated_features(corr_matrix, threshold)
        return X_in[reduced_corr_matrix.columns]
    
    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        self.drop_cols = []
        for col in X.columns:
            if col not in self.feature_target_rep:
                # FIXME: Currently, we assign best treatment outside of the CV. This sometimes leads to different features being filtered inside the CV. A solution could be less strong filters.
                continue
            if self.feature_target_rep[col] == 'raw':
                continue
            elif self.feature_target_rep[col] == 'mean':
                self.col_transformer[col] = Pipeline([
                    ('dtype', CategoricalDtypeAssigner()),
                    ('loo', LeaveOneOutEncoder()),
                    ('model', self.mean_transformer())
                ])
                self.col_transformer[col].fit(X[[col]], y)
            elif self.feature_target_rep[col] == 'linear':
                self.col_transformer[col] = self.linear_transformer()
                self.col_transformer[col].fit(X[[col]], y)
            elif 'combination_test' in self.feature_target_rep[col]:
                bins = int(self.feature_target_rep[col].split('_')[-1])
                self.col_transformer[col] = self.binner(bins)
                self.col_transformer[col] = Pipeline([
                    ('binner', self.binner(bins)),
                    ('model', self.mean_transformer()) # TODO: Test with LOO
                    ])
                self.col_transformer[col].fit(X[[col]], y)
            elif 'cat_threshold' in self.feature_target_rep[col]:
                bins = int(self.feature_target_rep[col].split('_')[-1])
                self.col_transformer[col] = self.mean_threshold_transformer(bins) # TODO: Test with LOO
                self.col_transformer[col].fit(X[[col]], y)
            elif self.feature_target_rep[col] == 'dummy':
                self.drop_cols.append(col)
            else:
                raise ValueError(f"Unknown feature_target_rep value: {self.feature_target_rep[col]} for column {col}")
            
            
            # if self.feature_target_rep[col] == 'mean':
            #     X[col] = self.col_transformer[col].transform(X[[col]])
            # else:
            X[col] = self.col_transformer[col].predict(X[[col]]) if self.target_type == 'regression' else self.col_transformer[col].predict_proba(X[[col]])[:,1]

        X_new = self.correlation_filter(X, threshold=0.95)  # Apply correlation filter
        
        self.drop_cols.extend(list(set(X.columns) - set(X_new.columns)))

        return self
    
    def transform(self, X_in):
        X = X_in.copy()
        X = X.drop(self.drop_cols, axis=1)
        for col in X.columns:
            if col in self.col_transformer:
                # if self.feature_target_rep[col] == 'mean':
                #     X[col] = self.col_transformer[col].transform(X[[col]])
                # else:
                X[col] = self.col_transformer[col].predict(X[[col]]) if self.target_type == 'regression' else self.col_transformer[col].predict_proba(X[[col]])[:,1]
        
        X = X.drop(columns=self.drop_cols, errors='ignore')
        return X

class FreqAdder(BaseEstimator, TransformerMixin):
    def __init__(self, candidate_cols=None):
        self.freq_maps = {}
        self.candidate_cols = candidate_cols
    
    def filter_candidates_by_distinctiveness(self, X):
            candidate_cols = []
            for col in X.columns:
                x_new = X[col].map(X[col].value_counts().to_dict())
                if all((pd.crosstab(X[col],x_new)>0).sum()==1):
                    continue
                else:
                    candidate_cols.append(col)

            return candidate_cols

    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        self.drop_cols = []

        if self.candidate_cols is None:
            self.candidate_cols = self.filter_candidates_by_distinctiveness(X)

        for col in self.candidate_cols:
            x = X[col]
            self.freq_maps[x.name] = x.value_counts().to_dict()

        return self
    
    def transform(self, X_in):
        X = X_in.copy()
        X = X.drop(self.drop_cols, axis=1)
        new_cols = []
        for col in X.columns:
            x = X[col]
            if x.name in self.freq_maps:
                new_col = x.map(self.freq_maps[x.name]).astype(float).fillna(0)
                new_col.name = x.name + "_freq"
                new_cols.append(new_col)
            else:
                continue

        return pd.concat([X]+new_cols, axis=1)
    
class CatIntAdder(BasePreprocessor):
    def __init__(self, 
                 target_type, use_filters=True,
                 max_order = 2, num_operations='all', 
                 candidate_cols=None,
                 add_freq=False, only_freq=False):
        super().__init__(target_type=target_type)
        self.use_filters = use_filters
        self.max_order = max_order
        self.num_operations = num_operations
        self.candidate_cols = candidate_cols
        self.add_freq = add_freq
        self.only_freq = only_freq

        self.new_dtypes = {}

    def combine(self, X_in, order=2, num_operations='all', seed=42, **kwargs):
        from itertools import combinations
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

    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        if self.candidate_cols is None:
            self.candidate_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.candidate_cols = [i for i in self.candidate_cols if X[i].nunique() > 5]  # TODO: Make this a parameter

        X_use = X[self.candidate_cols]
        X_new = pd.DataFrame(index=X_use.index)
        for order in range(2, self.max_order+1):
            X_new = pd.concat([X_new,
                self.combine(X_use, order=order, num_operations=self.num_operations)
            ], axis=1)

        self.new_col_set = list(set(X_new.columns)-set(X.columns))
        
        if self.use_filters:
            self.get_dummy_mean_scores(X_new[self.new_col_set], y)
            drop_cols = []
            target_mean_diff = {}
            for col in self.new_col_set:            
                if self.significances[col]['test_irrelevant_mean'] > self.alpha:
                    drop_cols.append(col)
                    self.new_col_set.remove(col)
                    X_new = X_new.drop(columns=drop_cols, errors='ignore')
                else:
                    loo_new = LeaveOneOutEncoder().fit_transform(X_new[[col]], y)
                    loo_old = LeaveOneOutEncoder().fit_transform(X_use[col.split('_&_')], y)
                    target_mean_diff[col] = loo_old.corrwith(loo_new[col], method='spearman').max()
                    if target_mean_diff[col] >0.95:
                        drop_cols.append(col)
                        self.new_col_set.remove(col)
                        X_new = X_new.drop(columns=drop_cols, errors='ignore')
            
            X_new = X_new.drop(columns=drop_cols, errors='ignore')
        X_new = pd.concat([X_use, X_new], axis=1)
    
        X_new = DropDuplicatesFeatureGenerator().fit_transform(X_new)

        self.new_col_set = list(set(X_new.columns)-set(X.columns))

        for col in self.new_col_set:
            # TODO: Might need to add option for NANs
            self.new_dtypes[col] = X_new[col].astype('category').dtype

        if self.add_freq or self.only_freq:
            # TODO: Unclear whether there is a more efficient way to do this
            cat_freq = FreqAdder()
            candidate_cols = cat_freq.filter_candidates_by_distinctiveness(X_new[self.new_col_set])
            if len(candidate_cols) > 0:
                self.cat_freq = FreqAdder(candidate_cols=candidate_cols).fit(X_new, y)        

        return self
    
    def transform(self, X_in):
        X = X_in.copy()

        X_out = X.copy()
        for degree in range(2, self.max_order+1):
            col_set_use = [col for col in self.new_col_set if col.count('_&_')+1 == degree]
            if len(col_set_use) > 0:
                X_degree = self.combine_predefined(X, col_set_use)
                for col in X_degree.columns:
                    X_degree[col] = X_degree[col].astype(self.new_dtypes[col])
                X_out = pd.concat([X_out, X_degree], axis=1)
                
        if self.add_freq or self.only_freq:
            X_out = self.cat_freq.transform(X_out)
        if self.only_freq:
            X_out = X_out.drop(self.new_col_set, axis=1, errors='ignore')

        return X_out
    

class CatGroupByAdder(BaseEstimator, TransformerMixin):
    def __init__(self, 
        candidate_cols=None,
        max_order = 2, num_operations='all', fillna=0):
        self.candidate_cols = candidate_cols
        self.max_order = max_order
        self.num_operations = num_operations
        self.fillna = fillna

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
            self.cnt_map[col1 + "_by_" + col2] = X[col1].groupby(X[col2]).nunique().fillna(self.fillna)
            one_by_two = X[col2].map(self.cnt_map[col1 + "_by_" + col2])
            one_by_two.name = col1 + "_by_" + col2
            new_cols.append(one_by_two)

            self.cnt_map[col2 + "_by_" + col1] = X[col2].groupby(X[col1]).nunique().fillna(self.fillna)
            two_by_one = X[col1].map(self.cnt_map[col2 + "_by_" + col1])
            two_by_one.name = col2 + "_by_" + col1
            new_cols.append(two_by_one)

        X = pd.concat(new_cols, axis=1)

        return X

    def groupby_predefined(self, X_in, comb_lst, **kwargs):
        X = X_in.copy()
        X_str = X.astype('U')
        feat_combs_use = [i.split("_by_") for i in comb_lst if "by" in i]
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

        new_cols = []
        for col1, col2 in zip(feat_combs_use_arr[0], feat_combs_use_arr[1]):
            one_by_two = X_str[col2].map(self.cnt_map[col1 + "_by_" + col2]).fillna(self.fillna)
            one_by_two.name = col1 + "_by_" + col2
            new_cols.append(one_by_two)

        X = pd.concat(new_cols, axis=1)
        return X

    def correlation_filter(self, X_in, threshold=0.95):
        corr_matrix = X_in.corr().abs()
        # Drop highly correlated features
        reduced_corr_matrix = drop_highly_correlated_features(corr_matrix, threshold)
        return X_in[reduced_corr_matrix.columns]

    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        
        if self.candidate_cols is None:
            self.candidate_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.candidate_cols = [i for i in self.candidate_cols if X[i].nunique() > 5]  # TODO: Make this a parameter

        X_new = self.groupby(X[self.candidate_cols], order=self.max_order, num_operations=self.num_operations)

        # Apply filters
        X_new = X_new.loc[: , (X_new.nunique()>1).values]
        X_new = self.correlation_filter(X_new, threshold=0.95)  # Apply correlation filter
        
        self.new_col_set = list(set(X_new.columns)-set(X.columns))
        
        return self
    
    def transform(self, X_in):
        X = X_in.copy()

        X_out = X.copy()
        for degree in range(2, self.max_order+1):
            col_set_use = [col for col in self.new_col_set if col.count('_by_')+1 == degree]
            if len(col_set_use) > 0:
                X_degree = self.groupby_predefined(X, col_set_use)
                X_out = pd.concat([X_out, X_degree], axis=1)

        return X_out


class CategoricalDtypeAssigner(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that:
    - Learns categories during fit (including NaN as its own category)
    - Converts to categorical with those categories during transform
    - Unseen values → NaN
    - NaN values → '__nan__' category
    """

    def __init__(self, columns=None, nan_token="__nan__"):
        self.columns = columns
        self.nan_token = nan_token
        self._categorical_dtypes = {}

    # def _replace_nan(self, series):
    #     return series.fillna(self.nan_token)

    def fit(self, X_in, y=None):
        X = X_in.copy()
        # cols = self.columns or X.select_dtypes(include=["object", "category"]).columns

        for col in X.columns:
            categories = X[col].unique().astype(str).tolist()+['nan']
            self._categorical_dtypes[col] = pd.CategoricalDtype(categories=categories, ordered=False)

        return self

    def transform(self, X_in):
        X = X_in.copy()

        for col, cat_dtype in self._categorical_dtypes.items():
            if col in X:
                X[col] = X[col].astype(str).astype(cat_dtype)

        return X
