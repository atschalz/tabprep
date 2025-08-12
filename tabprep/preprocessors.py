from __future__ import annotations
from typing import List, Dict

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, OrdinalEncoder, OneHotEncoder
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
from collections import Counter

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
        max_order = 2, num_operations='all', fillna=0,
       min_cardinality=6):
        self.candidate_cols = candidate_cols
        self.max_order = max_order
        self.num_operations = num_operations
        self.fillna = fillna
        self.min_cardinality = min_cardinality

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
            self.candidate_cols = [i for i in self.candidate_cols if X[i].nunique() >= self.min_cardinality]

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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, drop=None):
        """
        Parameters:
        -----------
        columns : list or None
            List of categorical column names to encode. 
            If None, all columns of type 'object' or 'category' will be encoded.
        drop : str or None
            If 'first', drop the first category to avoid collinearity.
        """
        self.columns = columns
        self.drop = drop
        self.categories_ = {}
    
    def fit(self, X, y=None):
        X = self._validate_input(X)
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.columns:
            self.categories_[col] = X[col].astype('category').cat.categories.tolist()
        return self

    def transform(self, X):
        X = self._validate_input(X)
        X_transformed = X.copy()
        for col in self.columns:
            dummies = pd.get_dummies(X_transformed[col], prefix=col)
            if self.drop == 'first':
                dummies = dummies.iloc[:, 1:]
            X_transformed = X_transformed.drop(col, axis=1)
            X_transformed = pd.concat([X_transformed, dummies], axis=1)
        return clean_feature_names(X_transformed)

    def _validate_input(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame or numpy ndarray.")
        return X

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, categories='auto', handle_unknown='error'):
        """
        Parameters:
        -----------
        columns : list or None
            List of categorical column names to encode. If None, auto-detect.
        categories : 'auto' or dict
            If 'auto', categories are inferred from training data.
            If dict, should map column names to list of categories.
        handle_unknown : str
            'error' (raise error on unknown category) or 'use_nan' (assign NaN).
        """
        self.columns = columns
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.category_maps_ = {}

    def fit(self, X, y=None):
        X = self._validate_input(X)

        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in self.columns:
            if isinstance(self.categories, dict) and col in self.categories:
                categories = self.categories[col]
            else:
                categories = X[col].astype('category').cat.categories.tolist()

            self.category_maps_[col] = {cat: i for i, cat in enumerate(categories)}

        return self

    def transform(self, X):
        X = self._validate_input(X)
        X_transformed = X.copy()

        for col in self.columns:
            mapping = self.category_maps_.get(col, {})
            if self.handle_unknown == 'error':
                unknowns = set(X_transformed[col]) - set(mapping.keys())
                if unknowns:
                    raise ValueError(f"Unknown categories {unknowns} found in column '{col}' during transform.")
            X_transformed[col] = X_transformed[col].map(mapping)
            if self.handle_unknown == 'use_nan':
                X_transformed[col] = X_transformed[col].astype('float')  # NaN is float
        return X_transformed

    def _validate_input(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame or numpy ndarray.")
        return X


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np

class SVDConcatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.svd = None  # will be initialized in fit

    def fit(self, X_in, y=None):
        X = X_in.copy()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                              unknown_value=-1)
        X[self.cat_cols] = self.ordinal_encoder.fit_transform(X[self.cat_cols])
        X_scaled = self.scaler.fit_transform(X)

        self.nan_filler = SimpleImputer(strategy='mean')
        X_scaled = self.nan_filler.fit_transform(X_scaled)


        n_components = max(
            1,
            min(
                int(X.shape[0] * 0.8) // 10 + 1,
                X.shape[1] // 2,
            ),
        )
        
        self.svd = TruncatedSVD(
            algorithm="arpack",
            n_components=n_components,
            random_state=self.random_state,
        )
        self.svd.fit(X_scaled)
        return self

    def transform(self, X_in):
        X = X_in.copy()
        X[self.cat_cols] = self.ordinal_encoder.transform(X[self.cat_cols])
        X_scaled = self.scaler.transform(X)
        X_scaled = self.nan_filler.transform(X_scaled)
        X_svd = pd.DataFrame(self.svd.transform(X_scaled), index=X.index)
        X_svd.columns = [f"svd_{i}" for i in range(X_svd.shape[1])]
        return pd.concat([X_in, X_svd], axis=1)

class CatAsNumTransformer(BaseEstimator, TransformerMixin):
    """
    Convert object/category columns:
      - If a column's non-null values are all numeric-like, cast it with pd.to_numeric.
      - Otherwise, apply an OrdinalEncoder (per-column) to map categories to integers.

    Parameters
    ----------
    handle_unknown : {"use_encoded_value", "error"}, default="use_encoded_value"
        Passed to each internal OrdinalEncoder.
    unknown_value : int or float, default=-1
        Value to use for unknown categories when handle_unknown="use_encoded_value".
    dtype : numpy dtype, default=np.float64
        Output dtype for numeric-like conversions and encoded columns.
    """
    def __init__(
        self,
        handle_unknown: str = "use_encoded_value",
        unknown_value: float | int = -1,
        dtype=np.float64,
    ):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatAsNumTransformer expects a pandas DataFrame input.")

        self.input_columns_: List[str] = list(X.columns)

        obj_cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_like_cols_: List[str] = []
        self.ordinal_cols_: List[str] = []
        self.encoders_: Dict[str, OrdinalEncoder] = {}

        # Decide column-by-column and fit encoders where needed
        for col in obj_cat_cols:
            s = X[col]
            # Determine if all non-null values can be parsed as numbers
            non_null = s.dropna()
            num_convertible = pd.to_numeric(non_null, errors="coerce").notna().all()

            if num_convertible:
                self.numeric_like_cols_.append(col)
            else:
                self.ordinal_cols_.append(col)
                enc = OrdinalEncoder(
                    handle_unknown=self.handle_unknown,
                    unknown_value=self.unknown_value,
                    dtype=self.dtype,
                )
                # Fit on a 2D array
                enc.fit(s.astype("category").to_frame())
                self.encoders_[col] = enc

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatAsNumTransformer expects a pandas DataFrame input.")

        # Keep original order/columns
        X_out = X.copy()

        # 1) Cast numeric-like string/category columns
        for col in self.numeric_like_cols_:
            if col in X_out.columns:
                X_out[col] = pd.to_numeric(X_out[col], errors="coerce").astype(self.dtype)

        # 2) Ordinal-encode the remaining categorical columns (per-column encoders)
        for col in self.ordinal_cols_:
            if col in X_out.columns:
                enc = self.encoders_[col]
                # transform expects 2D
                X_out[col] = enc.transform(X_out[[col]]).astype(self.dtype).ravel()

        return X_out

    # Feature names are unchanged (one-in/one-out)
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self, "input_columns_", None)
        if input_features is None:
            raise ValueError("Call fit before get_feature_names_out, or pass input_features.")
        return np.array(input_features, dtype=object)


import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_categorical_dtype

class CatOHETransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        handle_unknown: str = "ignore",
        sparse: bool = False,
        dtype=np.float64,
        min_frequency: int = 2
    ):
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.dtype = dtype
        self.min_frequency = int(min_frequency)

    # --- helpers ---
    def _ensure_tokens(self, s: pd.Series) -> pd.Series:
        """If categorical, add 'infrequent' and 'missing' categories before assignment."""
        if is_categorical_dtype(s):
            needed = [t for t in ("infrequent", "missing") if t not in s.cat.categories]
            if needed:
                s = s.cat.add_categories(needed)
        return s

    def _build_infrequent_map(self, X: pd.DataFrame) -> Dict[str, set]:
        infreq_map: Dict[str, set] = {}
        for col in self.cat_cols_:
            counts = X[col].value_counts(dropna=False)
            # mark values with count < min_frequency as infrequent; skip NaN (handled separately)
            infreq_map[col] = {v for v in counts.index if (not pd.isna(v)) and counts.loc[v] < self.min_frequency}
        return infreq_map

    def _apply_training_mapping(self, X: pd.DataFrame) -> pd.DataFrame:
        """Use TRAIN-time infrequent sets; NaN→'missing'; never recompute frequencies."""
        X = X.copy()
        for col in self.cat_cols_:
            s = X[col]
            s = self._ensure_tokens(s)
            # infrequent (based on train)
            if self.infrequent_values_[col]:
                mask = s.isin(self.infrequent_values_[col])
                # if s is object, .where works; if categorical, it's safe because tokens exist
                s = s.where(~mask, other="infrequent")
            # missing as its own category
            s = s.fillna("missing")
            X[col] = s
        return X

    # --- estimator API ---
    def fit(self, X: pd.DataFrame, y: Any = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatOHETransformer expects a pandas DataFrame input.")

        self.input_columns_: List[str] = list(X.columns)
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.non_cat_cols_ = [c for c in self.input_columns_ if c not in self.cat_cols_]

        self.infrequent_values_ = self._build_infrequent_map(X)

        X_cat_fit = self._apply_training_mapping(X[self.cat_cols_]) if self.cat_cols_ else pd.DataFrame(index=X.index)

        # sklearn >=1.2 uses sparse_output; older uses sparse
        try:
            self.ohe_ = OneHotEncoder(
                handle_unknown=self.handle_unknown,
                sparse_output=self.sparse,
                dtype=self.dtype
            )
        except TypeError:
            self.ohe_ = OneHotEncoder(
                handle_unknown=self.handle_unknown,
                sparse=self.sparse,
                dtype=self.dtype
            )

        if self.cat_cols_:
            self.ohe_.fit(X_cat_fit)

        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatOHETransformer expects a pandas DataFrame input.")

        if self.cat_cols_:
            X_cat = self._apply_training_mapping(X[self.cat_cols_])
            cat_encoded = self.ohe_.transform(X_cat)

            non_cat_cols_present = [c for c in self.non_cat_cols_ if c in X.columns]
            non_cat_df = X[non_cat_cols_present].copy()

            if not self.sparse:
                cat_df = pd.DataFrame(
                    cat_encoded,
                    columns=self.ohe_.get_feature_names_out(self.cat_cols_),
                    index=X.index
                )
                try:
                    cat_df = clean_feature_names(cat_df)  # if your helper exists
                except NameError:
                    pass
                return pd.concat([non_cat_df, cat_df], axis=1)
            else:
                # return a sparse design matrix with non-cats prepended
                from scipy import sparse as sp
                non_cat_mat = sp.csr_matrix(non_cat_df.to_numpy())
                return sp.hstack([non_cat_mat, cat_encoded], format="csr")

        # no categorical columns
        return X.copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "ohe_"):
            raise ValueError("Call fit before get_feature_names_out.")
        if self.cat_cols_:
            return np.array(
                list(self.non_cat_cols_) + list(self.ohe_.get_feature_names_out(self.cat_cols_)),
                dtype=object
            )
        return np.array(self.non_cat_cols_, dtype=object)

class CatLOOTransformer(BaseEstimator, TransformerMixin):
    """
    Transform all object/category columns using Leave-One-Out encoding.
    Keeps non-categorical columns unchanged and returns a pandas DataFrame.

    Parameters
    ----------
    sigma : float, default=0.0
        Standard deviation of Gaussian noise added to encoding.
    random_state : int, optional
        Random seed for noise.
    """
    def __init__(self, sigma: float = 0.0, random_state: int | None = None):
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatLOOTransformer expects a pandas DataFrame input.")
        if y is None:
            raise ValueError("Leave-One-Out encoding requires a target variable `y`.")

        self.input_columns_: List[str] = list(X.columns)

        # Identify categorical and non-categorical columns
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.non_cat_cols_ = [col for col in X.columns if col not in self.cat_cols_]

        # Fit Leave-One-Out encoder
        self.loo_ = LeaveOneOutEncoder(
            cols=self.cat_cols_,
            sigma=self.sigma,
            random_state=self.random_state
        )
        if self.cat_cols_:
            self.loo_.fit(X[self.cat_cols_], y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatLOOTransformer expects a pandas DataFrame input.")

        X_out = X.copy()

        if self.cat_cols_:
            cat_encoded = self.loo_.transform(X_out[self.cat_cols_])
            # Keep column names from encoder
            cat_encoded.columns = self.cat_cols_
            # Combine with non-categorical columns
            X_out = pd.concat([X_out[self.non_cat_cols_], cat_encoded], axis=1)

        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self, "input_columns_", None)
        if input_features is None:
            raise ValueError("Call fit before get_feature_names_out, or pass input_features.")
        return np.array(input_features, dtype=object)

class DuplicateCountAdder(BaseEstimator, TransformerMixin):
    """
    A transformer that counts duplicate samples in the training set
    and appends a new feature with those counts at transform time.
    """

    def __init__(self, feature_name: str = "duplicate_count", cap=False):
        self.feature_name = feature_name
        self.cap = cap

    def fit(self, X_in, y=None):
        """
        Learn the duplicate‐count mapping from X.
        
        Parameters
        ----------
        X : array‐like or DataFrame, shape (n_samples, n_features)
            Training data.

        y : Ignored.
        
        Returns
        -------
        self
        """
        X = X_in.copy()
        # If DataFrame, extract values but remember columns
        if isinstance(X, pd.DataFrame):
            self._is_df = True
            self._columns_ = X.columns.tolist()
            data = X.values
        else:
            self._is_df = False
            data = np.asarray(X)
        
        # Convert each row into a hashable tuple and count
        rows = [tuple(row) for row in data]
        self.counts_ = Counter(rows)
        return self

    def transform(self, X_in):
        """
        Append the duplicate‐count feature to X.
        
        Parameters
        ----------
        X : array‐like or DataFrame, shape (n_samples, n_features)
            New data to transform.
        
        Returns
        -------
        X_out : same type as X but with one extra column
        """
        X = X_in.copy()
        # Prepare raw array and remember if DataFrame
        if isinstance(X, pd.DataFrame):
            cols = X.columns.tolist()
            data = X.values
        else:
            data = np.asarray(X)
        
        # Look up counts (0 if unseen)
        new_feat = pd.Series([self.counts_.get(tuple(row), 0) for row in data], index=X.index, name=self.feature_name)
        
        if self.cap:
            new_feat = new_feat.clip(upper=1)

        out = pd.concat([X, new_feat], axis=1)
        return out


from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DuplicateContentLOOEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, new_name='LOO_duplicates'
    ):
        self.new_col = new_name

    def fit(self, X, y):
        X_str = X.astype(str).sum(axis=1)

        self.u_id_map = {i:j for i,j in zip(X_str.unique(),list(range(X_str.nunique())))}

        x_uid = X_str.map(self.u_id_map).astype(str)
        self.loo = LeaveOneOutEncoder()
        self.loo.fit(x_uid, y)

        return self

    def transform(self, X):
        X_out = X.copy()
        X_str = X.astype(str).sum(axis=1)
        x_uid = X_str.map(self.u_id_map).astype(str)
        X_out[self.new_col] = self.loo.transform(x_uid)
        return X_out
    

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from inspect import signature

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

@dataclass
class _CatSpec:
    frequent_values: set

class LinearFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Pandas in/out. Steps:
      - Numeric: mean-impute + add {col}_missing indicator
      - Categorical: bucket rare (<min_count) to rare_token; OHE
      - Fit LinearRegression/LogisticRegression on encoded matrix
      - Append prediction columns to X (and optionally append encoded features)

    Parameters
    ----------
    model_params : dict
    prefix : str
    use_proba_for_classification : bool
    min_count : int
    rare_token : str
    categorical_cols : list[str] or None
    append_encoded : bool
        If True, append encoded features (numeric imputed + indicators + OHE) to output.
    """
    def __init__(
        self,
        target_type: str,
        linear_model_type: str = 'default',
        model_params: Optional[Dict[str, Any]] = None,
        prefix: str = "lin",
        use_proba_for_classification: bool = True,
        min_count: int = 5,
        rare_token: str = "__OTHER__",
        categorical_cols: Optional[List[str]] = None,
        append_encoded: bool = False,
        random_state: Optional[int] = 42
    ):
        self.target_type = target_type
        self.linear_model_type = linear_model_type
        self.model_params = model_params or {}
        self.prefix = prefix
        self.use_proba_for_classification = use_proba_for_classification
        self.min_count = min_count
        self.rare_token = rare_token
        self.categorical_cols = categorical_cols
        self.append_encoded = append_encoded
        self.random_state = random_state


        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # TODO: Extend to Ridge/Lasso/ElasticNet
        if self.linear_model_type == 'default':
            if self.target_type == "regression":
                self.model_ = LinearRegression(**self.model_params, random_state=self.random_state)
            elif self.target_type in ("binary", "multiclass"):
                self.model_ = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=self.random_state)
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")
        elif self.linear_model_type == 'lasso':
            if self.target_type == "regression":
                self.model_ = Lasso(**self.model_params, random_state=self.random_state)
            elif self.target_type in ("binary", "multiclass"):
                self.model_ = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000, random_state=self.random_state)
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")
        elif self.linear_model_type == 'ridge':
            if self.target_type == "regression":
                self.model_ = Ridge(**self.model_params, random_state=self.random_state)
            elif self.target_type in ("binary", "multiclass"):
                self.model_ = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=self.random_state)
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")
        elif self.linear_model_type == 'elasticnet':
            if self.target_type == "regression":
                self.model_ = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.random_state)
            elif self.target_type in ("binary", "multiclass"):
                self.model_ = LogisticRegression(penalty='elasticnet', l1_ratio=0.5,
                             C=1.0, solver='saga', max_iter=1000, random_state=self.random_state)
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")
        else:
            raise ValueError(f"Unsupported linear model type: {self.linear_model_type}")

    # ---------- Helpers ----------
    def _ensure_df(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _ensure_series(self, y, index):
        return y if isinstance(y, pd.Series) else pd.Series(y, index=index)

    def _auto_cats(self, X: pd.DataFrame) -> List[str]:
        cats = []
        for c in X.columns:
            dt = X[c].dtype
            if (pd.api.types.is_object_dtype(dt)
                or isinstance(dt, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(dt)):
                cats.append(c)
        if self.categorical_cols is not None:
            cats = [c for c in self.categorical_cols if c in X.columns]
        return cats

    def _rare_map(self, s: pd.Series, spec: _CatSpec) -> pd.Series:
        return s.where(s.isin(spec.frequent_values), other=self.rare_token).fillna(self.rare_token)

    def _make_ohe(self):
        # sklearn >=1.4 removed 'sparse' in favor of 'sparse_output'
        params = signature(OneHotEncoder).parameters
        if "sparse_output" in params:
            return OneHotEncoder(handle_unknown="ignore", dtype=float, sparse_output=False)
        else:
            return OneHotEncoder(handle_unknown="ignore", dtype=float, sparse=False)

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build the design matrix used by the internal linear/logistic model."""
        idx = X.index

        # --- numeric: missing indicator + mean impute ---
        if self.num_cols_:
            # missing indicators from *current* X before imputation
            miss_ind = X[self.num_cols_].isna().astype(int)
            miss_ind.columns = [f"{c}_missing" for c in self.num_cols_]

            # impute with training means
            X_num = X[self.num_cols_].copy()
            for c in self.num_cols_:
                X_num[c] = X_num[c].astype("float64")
                X_num[c] = X_num[c].fillna(self.num_impute_values_[c])
        else:
            miss_ind = pd.DataFrame(index=idx)
            X_num = pd.DataFrame(index=idx)

        # --- categoricals: rare-bucket + OHE ---
        if self.cat_cols_:
            mapped = pd.DataFrame({
                c: self._rare_map(X[c].astype("object"), self.cat_specs_[c])
                for c in self.cat_cols_
            }, index=idx)
            enc = self.ohe_.transform(mapped)
            ohe_df = pd.DataFrame(enc, index=idx, columns=self.ohe_feature_names_) if enc.size else pd.DataFrame(index=idx)
        else:
            ohe_df = pd.DataFrame(index=idx)

        # concat in stable order: numeric -> indicators -> ohe
        X_enc = pd.concat([X_num, miss_ind, ohe_df], axis=1)
        
        return self.X_scaler.fit_transform(X_enc)

    def _encode_unseen(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build the design matrix used by the internal linear/logistic model."""
        idx = X.index

        # --- numeric: missing indicator + mean impute ---
        if self.num_cols_:
            # missing indicators from *current* X before imputation
            miss_ind = X[self.num_cols_].isna().astype(int)
            miss_ind.columns = [f"{c}_missing" for c in self.num_cols_]

            # impute with training means
            X_num = X[self.num_cols_].copy()
            for c in self.num_cols_:
                X_num[c] = X_num[c].astype("float64")
                X_num[c] = X_num[c].fillna(self.num_impute_values_[c])
        else:
            miss_ind = pd.DataFrame(index=idx)
            X_num = pd.DataFrame(index=idx)

        # --- categoricals: rare-bucket + OHE ---
        if self.cat_cols_:
            mapped = pd.DataFrame({
                c: self._rare_map(X[c].astype("object"), self.cat_specs_[c])
                for c in self.cat_cols_
            }, index=idx)
            enc = self.ohe_.transform(mapped)
            ohe_df = pd.DataFrame(enc, index=idx, columns=self.ohe_feature_names_) if enc.size else pd.DataFrame(index=idx)
        else:
            ohe_df = pd.DataFrame(index=idx)

        # concat in stable order: numeric -> indicators -> ohe
        X_enc = pd.concat([X_num, miss_ind, ohe_df], axis=1)
        
        return self.X_scaler.transform(X_enc)



    # ---------- sklearn API ----------
    def fit(self, X, y):
        X = self._ensure_df(X)
        y = self._ensure_series(y, index=X.index)

        if self.target_type == "regression":
            y = pd.Series(self.y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel(), index=X.index, name=y.name)

        self.feature_names_in_ = list(X.columns)
        self.cat_cols_ = self._auto_cats(X)
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        # numeric means for imputation (computed on training set)
        self.num_impute_values_ = {c: pd.to_numeric(X[c], errors="coerce").mean() for c in self.num_cols_}

        # rare specs for categoricals
        self.cat_specs_ = {}
        for c in self.cat_cols_:
            vc = pd.Series(X[c]).astype("object").value_counts(dropna=False)
            frequent = set(vc[vc >= self.min_count].index)
            self.cat_specs_[c] = _CatSpec(frequent_values=frequent)

        # fit OHE on mapped categoricals
        # TODO: Use ordinal-encoding for high-cardinality categoricals
        if self.cat_cols_:
            mapped = pd.DataFrame({
                c: self._rare_map(pd.Series(X[c]).astype("object"), self.cat_specs_[c])
                for c in self.cat_cols_
            }, index=X.index)
            self.ohe_ = self._make_ohe().fit(mapped)
            self.ohe_feature_names_ = self.ohe_.get_feature_names_out(self.cat_cols_).tolist()
        else:
            self.ohe_ = None
            self.ohe_feature_names_ = []

        # names for encoded output (when append_encoded=True)
        self.missing_indicator_names_ = [f"{c}_missing" for c in self.num_cols_]
        self.encoded_feature_names_ = self.num_cols_ + self.missing_indicator_names_ + self.ohe_feature_names_

        # design matrix & model
        X_enc = self._encode(X)

        self.model_.fit(X_enc, y)
        if self.target_type in ("binary", "multiclass"):
            self.classes_ = self.model_.classes_

        return self

    def transform(self, X):
        X = self._ensure_df(X)

        X_enc = self._encode_unseen(X)

        # predictions
        if self.target_type == "regression":
            preds = self.model_.predict(X_enc)
            add = pd.DataFrame({f"{self.prefix}_pred": preds}, index=X.index)

        elif self.target_type == "binary":
            if self.use_proba_for_classification:
                pos_label = self.classes_[1] if len(self.classes_) >= 2 else self.classes_[0]
                proba = self.model_.predict_proba(X_enc)[:, list(self.classes_).index(pos_label)]
                add = pd.DataFrame({f"{self.prefix}_proba_{pos_label}": proba}, index=X.index)
            else:
                labels = self.model_.predict(X_enc)
                add = pd.DataFrame({f"{self.prefix}_label": labels}, index=X.index)

        elif self.target_type == "multiclass":
            if self.use_proba_for_classification:
                proba = self.model_.predict_proba(X_enc)
                cols = [f"{self.prefix}_proba_{c}" for c in self.classes_]
                add = pd.DataFrame(proba, index=X.index, columns=cols)
            else:
                labels = self.model_.predict(X_enc)
                add = pd.DataFrame({f"{self.prefix}_label": labels}, index=X.index)
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")

        if self.append_encoded:
            enc_out = X_enc.copy()
            enc_out.columns = self.encoded_feature_names_
            return pd.concat([X, enc_out, add], axis=1)

        return pd.concat([X, add], axis=1)

    def get_feature_names_out(self, input_features=None):
        base = self.feature_names_in_
        if self.append_encoded:
            base = base + self.encoded_feature_names_

        if self.target_type == "continuous":
            extra = [f"{self.prefix}_pred"]
        elif self.target_type == "binary":
            extra = ([f"{self.prefix}_label"] if not self.use_proba_for_classification
                     else [f"{self.prefix}_proba_{self.classes_[1] if len(self.classes_)>=2 else self.classes_[0]}"])
        else:
            extra = ([f"{self.prefix}_label"] if not self.use_proba_for_classification
                     else [f"{self.prefix}_proba_{c}" for c in self.classes_])

        return np.array(base + extra)


from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, StratifiedKFold

from inspect import signature

@dataclass
class _CatSpec:
    frequent_values: set

class OOFLinearFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Pandas in/out. Steps:
      - Numeric: mean-impute + add {col}_missing indicator
      - Categorical: bucket rare (<min_count) to rare_token; OHE
      - Fit linear/logistic models using K-fold CV; store OOF preds
      - Append OOF preds on fit_transform; average-fold preds at transform
      - (Optional) append encoded features

    Parameters
    ----------
    target_type : {'regression','binary','multiclass'}
    linear_model_type : {'default','lasso','ridge','elasticnet'}
    model_params : dict
    prefix : str
    use_proba_for_classification : bool
    min_count : int
    rare_token : str
    categorical_cols : list[str] or None
    append_encoded : bool
    n_splits : int
    cv_shuffle : bool
    cv_random_state : int | None
    """
    def __init__(
        self,
        target_type: str,
        linear_model_type: str = 'default',
        model_params: Optional[Dict[str, Any]] = None,
        prefix: str = "lin",
        use_proba_for_classification: bool = True,
        min_count: int = 5,
        rare_token: str = "__OTHER__",
        categorical_cols: Optional[List[str]] = None,
        append_encoded: bool = False,
        n_splits: int = 5,
        cv_shuffle: bool = True,
        cv_random_state: Optional[int] = 42,
    ):
        self.target_type = target_type
        self.linear_model_type = linear_model_type
        self.model_params = model_params or {}
        self.prefix = prefix
        self.use_proba_for_classification = use_proba_for_classification
        self.min_count = min_count
        self.rare_token = rare_token
        self.categorical_cols = categorical_cols
        self.append_encoded = append_encoded
        self.n_splits = n_splits
        self.cv_shuffle = cv_shuffle
        self.cv_random_state = cv_random_state

        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # set up a prototype model; actual folds will use clones
        self.model_ = self._make_model()

    # ---------- Helpers ----------
    def _make_model(self):
        if self.linear_model_type == 'default':
            if self.target_type == "regression":
                return LinearRegression(**self.model_params, random_state=self.cv_random_state)
            elif self.target_type in ("binary", "multiclass"):
                # penalty=None is only valid for some solvers/versions; keep simple/robust
                return LogisticRegression(
                    penalty='l2', solver='lbfgs', max_iter=1000, **self.model_params, random_state=self.cv_random_state
                )
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")
        elif self.linear_model_type == 'lasso':
            if self.target_type == "regression":
                return Lasso(**self.model_params, random_state=self.cv_random_state)
            elif self.target_type in ("binary", "multiclass"):
                return LogisticRegression(
                    penalty='l1', C=1.0, solver='liblinear', max_iter=1000, **self.model_params, random_state=self.cv_random_state
                )
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")
        elif self.linear_model_type == 'ridge':
            if self.target_type == "regression":
                return Ridge(**self.model_params, random_state=self.cv_random_state)
            elif self.target_type in ("binary", "multiclass"):
                return LogisticRegression(
                    penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, **self.model_params, random_state=self.cv_random_state
                )
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")
        elif self.linear_model_type == 'elasticnet':
            if self.target_type == "regression":
                return ElasticNet(alpha=1.0, l1_ratio=0.5, **self.model_params, random_state=self.cv_random_state)
            elif self.target_type in ("binary", "multiclass"):
                return LogisticRegression(
                    penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga', max_iter=1000, **self.model_params, random_state=self.cv_random_state
                )
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")
        else:
            raise ValueError(f"Unsupported linear model type: {self.linear_model_type}")

    def _ensure_df(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _ensure_series(self, y, index):
        return y if isinstance(y, pd.Series) else pd.Series(y, index=index)

    def _auto_cats(self, X: pd.DataFrame) -> List[str]:
        cats = []
        for c in X.columns:
            dt = X[c].dtype
            if (pd.api.types.is_object_dtype(dt)
                or isinstance(dt, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(dt)):
                cats.append(c)
        if self.categorical_cols is not None:
            cats = [c for c in self.categorical_cols if c in X.columns]
        return cats

    def _rare_map(self, s: pd.Series, spec: _CatSpec) -> pd.Series:
        return s.where(s.isin(spec.frequent_values), other=self.rare_token).fillna(self.rare_token)

    def _make_ohe(self):
        params = signature(OneHotEncoder).parameters
        if "sparse_output" in params:
            return OneHotEncoder(handle_unknown="ignore", dtype=float, sparse_output=False)
        else:
            return OneHotEncoder(handle_unknown="ignore", dtype=float, sparse=False)

    def _encode_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build raw (unscaled) design matrix used by the internal models."""
        idx = X.index

        # --- numeric: missing indicator + mean impute ---
        if self.num_cols_:
            miss_ind = X[self.num_cols_].isna().astype(int)
            miss_ind.columns = [f"{c}_missing" for c in self.num_cols_]

            X_num = X[self.num_cols_].copy()
            for c in self.num_cols_:
                X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
                X_num[c] = X_num[c].fillna(self.num_impute_values_[c])
        else:
            miss_ind = pd.DataFrame(index=idx)
            X_num = pd.DataFrame(index=idx)

        # --- categoricals: rare-bucket + OHE ---
        if self.cat_cols_:
            mapped = pd.DataFrame({
                c: self._rare_map(X[c].astype("object"), self.cat_specs_[c])
                for c in self.cat_cols_
            }, index=idx)
            enc = self.ohe_.transform(mapped)
            ohe_df = pd.DataFrame(enc, index=idx, columns=self.ohe_feature_names_) if enc.size else pd.DataFrame(index=idx)
        else:
            ohe_df = pd.DataFrame(index=idx)

        # concat in stable order: numeric -> indicators -> ohe
        X_enc = pd.concat([X_num, miss_ind, ohe_df], axis=1)
        return X_enc

    def _scale(self, Xdf: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if fit:
            arr = self.X_scaler.fit_transform(Xdf.values)
        else:
            arr = self.X_scaler.transform(Xdf.values)
        return pd.DataFrame(arr, index=Xdf.index, columns=Xdf.columns)

    # ---------- sklearn API ----------
    def fit(self, X, y):
        X = self._ensure_df(X)
        y = self._ensure_series(y, index=X.index)

        if self.target_type == "regression":
            y = pd.Series(self.y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel(), index=X.index, name=y.name)

        self.feature_names_in_ = list(X.columns)
        self.cat_cols_ = self._auto_cats(X)
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        # numeric means for imputation (computed on training set)
        self.num_impute_values_ = {c: pd.to_numeric(X[c], errors="coerce").mean() for c in self.num_cols_}

        # rare specs for categoricals
        self.cat_specs_ = {}
        for c in self.cat_cols_:
            vc = pd.Series(X[c]).astype("object").value_counts(dropna=False)
            frequent = set(vc[vc >= self.min_count].index)
            self.cat_specs_[c] = _CatSpec(frequent_values=frequent)

        # fit OHE on mapped categoricals
        if self.cat_cols_:
            mapped = pd.DataFrame({
                c: self._rare_map(pd.Series(X[c]).astype("object"), self.cat_specs_[c])
                for c in self.cat_cols_
            }, index=X.index)
            self.ohe_ = self._make_ohe().fit(mapped)
            self.ohe_feature_names_ = self.ohe_.get_feature_names_out(self.cat_cols_).tolist()
        else:
            self.ohe_ = None
            self.ohe_feature_names_ = []

        # names for encoded output (when append_encoded=True)
        self.missing_indicator_names_ = [f"{c}_missing" for c in self.num_cols_]
        self.encoded_feature_names_ = self.num_cols_ + self.missing_indicator_names_ + self.ohe_feature_names_

        # raw encode + fit scaler once
        X_raw = self._encode_raw(X)
        X_enc = self._scale(X_raw, fit=True)

        # set up CV splitter
        if self.target_type == "regression":
            splitter = KFold(n_splits=self.n_splits, shuffle=self.cv_shuffle, random_state=self.cv_random_state)
        else:
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=self.cv_shuffle, random_state=self.cv_random_state)

        n = len(X)
        self.models_ = []
        self.classes_ = None

        # allocate OOF container
        if self.target_type == "regression":
            oof = np.zeros(n, dtype=float)
        elif self.target_type == "binary":
            oof = np.zeros(n, dtype=float) if self.use_proba_for_classification else np.empty(n, dtype=object)
        else:  # multiclass
            # We'll get classes_ after first fold fit; temporarily defer shape
            oof = None

        for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_enc, y if self.target_type=="regression" else y.astype(str))):
            model = clone(self._make_model())
            X_tr, X_va = X_enc.iloc[tr_idx], X_enc.iloc[va_idx]
            y_tr = y.iloc[tr_idx]

            model.fit(X_tr, y_tr)
            self.models_.append(model)

            # Establish classes_ and oof shape for multiclass on first fold
            if self.target_type in ("binary","multiclass") and self.classes_ is None:
                self.classes_ = model.classes_
                if self.target_type == "multiclass":
                    oof = np.zeros((n, len(self.classes_)), dtype=float)

            # collect OOF predictions
            if self.target_type == "regression":
                pred = model.predict(X_va)
                oof[va_idx] = pred
            elif self.target_type == "binary":
                if self.use_proba_for_classification:
                    pos_label = self.classes_[1] if len(self.classes_) >= 2 else self.classes_[0]
                    proba = model.predict_proba(X_va)[:, list(self.classes_).index(pos_label)]
                    oof[va_idx] = proba
                else:
                    labels = model.predict(X_va)
                    oof[va_idx] = labels
            else:  # multiclass
                if self.use_proba_for_classification:
                    proba = model.predict_proba(X_va)
                    oof[va_idx, :] = proba
                else:
                    labels = model.predict(X_va)
                    # convert labels to columns in class order
                    tmp = np.zeros((len(va_idx), len(self.classes_)), dtype=float)
                    for i, lab in enumerate(labels):
                        tmp[i, list(self.classes_).index(lab)] = 1.0
                    oof[va_idx, :] = tmp

        # store OOF predictions as DataFrame aligned to X.index
        if self.target_type == "regression":
            self.oof_predictions_ = pd.DataFrame({f"{self.prefix}_pred": oof}, index=X.index)
        elif self.target_type == "binary":
            if self.use_proba_for_classification:
                pos_label = self.classes_[1] if len(self.classes_) >= 2 else self.classes_[0]
                self.oof_predictions_ = pd.DataFrame({f"{self.prefix}_proba_{pos_label}": oof}, index=X.index)
            else:
                self.oof_predictions_ = pd.DataFrame({f"{self.prefix}_label": oof}, index=X.index)
        else:
            if self.use_proba_for_classification:
                cols = [f"{self.prefix}_proba_{c}" for c in self.classes_]
                self.oof_predictions_ = pd.DataFrame(oof, index=X.index, columns=cols)
            else:
                cols = [f"{self.prefix}_label_{c}" for c in self.classes_]
                self.oof_predictions_ = pd.DataFrame(oof, index=X.index, columns=cols)

        return self

    def _predict_from_models(self, X_enc: pd.DataFrame) -> pd.DataFrame:
        """Average predictions across fold models for inference."""
        if self.target_type == "regression":
            preds = np.stack([m.predict(X_enc) for m in self.models_], axis=1).mean(axis=1)
            return pd.DataFrame({f"{self.prefix}_pred": preds}, index=X_enc.index)

        elif self.target_type == "binary":
            if self.use_proba_for_classification:
                pos_label = self.classes_[1] if len(self.classes_) >= 2 else self.classes_[0]
                idx = list(self.classes_).index(pos_label)
                probs = np.stack([m.predict_proba(X_enc)[:, idx] for m in self.models_], axis=1).mean(axis=1)
                return pd.DataFrame({f"{self.prefix}_proba_{pos_label}": probs}, index=X_enc.index)
            else:
                # majority vote
                labels = np.stack([m.predict(X_enc) for m in self.models_], axis=1)
                # simple mode
                out = pd.Series([pd.Series(row).mode().iloc[0] for row in labels], index=X_enc.index)
                return pd.DataFrame({f"{self.prefix}_label": out}, index=X_enc.index)

        else:  # multiclass
            if self.use_proba_for_classification:
                prob_stacks = [m.predict_proba(X_enc) for m in self.models_]
                avg = np.stack(prob_stacks, axis=2).mean(axis=2)
                cols = [f"{self.prefix}_proba_{c}" for c in self.classes_]
                return pd.DataFrame(avg, index=X_enc.index, columns=cols)
            else:
                # majority vote -> one label column
                labels = np.stack([m.predict(X_enc) for m in self.models_], axis=1)
                out = pd.Series([pd.Series(row).mode().iloc[0] for row in labels], index=X_enc.index)
                return pd.DataFrame({f"{self.prefix}_label": out}, index=X_enc.index)

    def transform(self, X):
        X = self._ensure_df(X)

        # encode with already-fit encoders, then scale with already-fit scaler
        X_raw = self._encode_raw(X)
        X_enc = self._scale(X_raw, fit=False)

        # averaged predictions across folds
        add = self._predict_from_models(X_enc)

        if self.append_encoded:
            enc_out = X_enc.copy()
            enc_out.columns = self.encoded_feature_names_
            return pd.concat([X, enc_out, add], axis=1)

        return pd.concat([X, add], axis=1)

    def fit_transform(self, X, y):
        """Return X with **OOF predictions** appended (not averaged predictions)."""
        self.fit(X, y)
        add = self.oof_predictions_
        if self.append_encoded:
            X_raw = self._encode_raw(self._ensure_df(X))
            X_enc = self._scale(X_raw, fit=False)
            enc_out = X_enc.copy()
            enc_out.columns = self.encoded_feature_names_
            return pd.concat([self._ensure_df(X), enc_out, add], axis=1)
        return pd.concat([self._ensure_df(X), add], axis=1)

    def get_feature_names_out(self, input_features=None):
        base = self.feature_names_in_
        if self.append_encoded:
            base = base + self.encoded_feature_names_

        if self.target_type == "regression":
            extra = [f"{self.prefix}_pred"]
        elif self.target_type == "binary":
            extra = ([f"{self.prefix}_label"] if not self.use_proba_for_classification
                     else [f"{self.prefix}_proba_{self.classes_[1] if len(self.classes_)>=2 else self.classes_[0]}"])
        else:
            extra = ([f"{self.prefix}_label"] if not self.use_proba_for_classification
                     else [f"{self.prefix}_proba_{c}" for c in self.classes_])

        return np.array(base + extra)


from tabprep.proxy_models import CustomLinearModel
class ModelLinearFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, target_type):
        self.target_type = target_type
        self.model = CustomLinearModel(target_type=target_type)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        X_out = X.copy()
        if self.target_type == "regression":
            X_out['linear'] = self.model.predict(X)
        elif self.target_type == "binary":
            X_out['linear'] = self.model.predict(X)
        elif self.target_type == "multiclass":
            preds = self.model.predict(X)
            for i, label in enumerate(self.model.classes_):
                X_out[f'linear_{label}'] = preds[:, i]
        return X_out
