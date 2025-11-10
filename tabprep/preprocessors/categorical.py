import numpy as np
import pandas as pd
from tabprep.detectors.base_preprocessor import BasePreprocessor as old_base
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.preprocessing import TargetEncoder
from tabprep.preprocessors.frequency import FrequencyEncoder
from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator
from tabprep.utils.misc import sample_from_set
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from tabprep.utils.misc import drop_highly_correlated_features
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_categorical_dtype
from tabprep.utils.modeling_utils import clean_feature_names

from tabprep.preprocessors.base import CategoricalBasePreprocessor

from typing import List, Dict, Any, Literal
from sklearn.model_selection import KFold, StratifiedKFold

# TODO: Adjust to new preprocessing logic
# TODO: Add duplicate filter (either using names or factorize after generation)
class CatIntAdder(old_base):
    def __init__(self, 
                 target_type, 
                 use_filters=True,
                 max_order = 2, 
                 num_operations='all', 
                 candidate_cols=None,
                 add_freq=False, 
                 only_freq=False,
                 min_cardinality=6,
                 fillna: int = 0,
                 log: bool = False,
                 **kwargs
                 ):
        super().__init__(target_type=target_type)

        self.__name__ = "CatIntAdder"
        self.use_filters = use_filters
        self.max_order = max_order
        self.num_operations = num_operations
        self.candidate_cols = candidate_cols
        self.add_freq = add_freq
        self.only_freq = only_freq
        self.min_cardinality = min_cardinality
        self.fillna = fillna
        self.log = log

        self.new_dtypes = {}

        del self.cv_func, self.adjust_target_format # NOTE: Necessary to store AG models with pickle

    def combine(self, X_in, order=2, num_operations='all', seed=42, **kwargs):
        # TODO: Implement as matrix operations to speed up the process
        X = X_in.copy()
        X = X.astype('U')
        feat_combs_use = list(combinations(np.unique(X.columns), order))
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

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
            self.candidate_cols = [i for i in self.candidate_cols if X[i].nunique() >= self.min_cardinality]  # TODO: Make this a parameter

        if len(self.candidate_cols) < self.max_order:
            self.new_col_set = []
            return self

        X_use = X[self.candidate_cols]
        X_new = pd.DataFrame(index=X_use.index)
        for order in range(2, self.max_order+1):
            X_new = pd.concat([X_new,
                self.combine(X_use, order=order, num_operations=self.num_operations)
            ], axis=1)

        self.new_col_set = [c for c in X_new.columns if c not in X.columns]
        
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

        self.new_col_set = [c for c in X_new.columns if c not in X.columns]

        for col in self.new_col_set:
            # TODO: Might need to add option for NANs
            self.new_dtypes[col] = X_new[col].astype('category').dtype

        if self.add_freq or self.only_freq:
            # TODO: Unclear whether there is a more efficient way to do this
            cat_freq = FrequencyEncoder(fillna=self.fillna, log=self.log)
            candidate_cols = cat_freq.filter_candidates_by_distinctiveness(X_new[self.new_col_set])
            if len(candidate_cols) > 0:
                self.cat_freq = FrequencyEncoder(candidate_cols=candidate_cols, fillna=self.fillna, log=self.log).fit(X_new[candidate_cols], y)

        return self
    
    def transform(self, X_in, **kwargs):
        X = X_in.copy()

        X_out = pd.DataFrame(index=X.index)
        if len(self.new_col_set) > 0:
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
        X_out = pd.concat([X, X_out], axis=1)
        return X_out

# class Cat

# TODO: Adjust to new preprocessinbg logic
class CatGroupByAdder(BaseEstimator, TransformerMixin):
    def __init__(self, 
        candidate_cols=None,
        max_order = 2, num_operations='all', fillna=0,
       min_cardinality=6,
       **kwargs
       ):
        self.candidate_cols = candidate_cols
        self.max_order = max_order
        self.num_operations = num_operations
        self.fillna = fillna
        self.min_cardinality = min_cardinality
        self.__name__ = "CatGroupByAdder"

    def groupby(self, X_in, order=2, num_operations='all', seed=42, **kwargs):
        # TODO: Implement as matrix operations to speed up the process
        self.cnt_map = {}
        X = X_in.copy()
        X = X.astype('U')
        feat_combs_use = list(combinations(np.unique(X.columns), order))
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

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

        if len(self.candidate_cols) < 2:
            self.new_col_set = []
            return self

        X_new = self.groupby(X[self.candidate_cols], order=self.max_order, num_operations=self.num_operations)

        # Apply filters
        X_new = X_new.loc[: , (X_new.nunique()>1).values]
        X_new = self.correlation_filter(X_new, threshold=0.95)  # Apply correlation filter
        
        self.new_col_set = [c for c in X_new.columns if c not in X.columns]
        
        return self
    
    def transform(self, X_in, **kwargs):
        X = X_in.copy()

        X_out = X.copy() # pd.DataFrame(index=X.index)
        for degree in range(2, self.max_order+1):
            col_set_use = [col for col in self.new_col_set if col.count('_by_')+1 == degree]
            if len(col_set_use) > 0:
                X_degree = self.groupby_predefined(X, col_set_use)
                X_out = pd.concat([X_out, X_degree], axis=1)

        return X_out
    
class OneHotPreprocessor(CategoricalBasePreprocessor):
    """
    One-hot encode categorical columns with one level dropped per feature (to avoid multicollinearity).
    - Column naming: <original_col>__<category_value>
    - Unknown categories at transform time are ignored (all-zeros for that feature).
    """

    def __init__(
        self, 
        keep_original: bool = False, 
        drop: Literal['if_binary', 'first'] = 'if_binary', 
        min_frequency: int = 1,
        **kwargs

        ):
        super().__init__(keep_original=keep_original)
        self.drop = drop
        self.min_frequency = min_frequency

    def _fit(self, X: pd.DataFrame, y=None):
        # categories = [pd.Series(col).unique().tolist() for _, col in X.items()]
        self.encoder_ = OneHotEncoder(
            # categories=categories,
            drop=self.drop,
            handle_unknown="ignore",
            sparse_output=False,
        ).fit(X)
        self.feature_names_ = self.encoder_.get_feature_names_out(X.columns)
        return self

    def _transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        arr = self.encoder_.transform(X)
        return pd.DataFrame(arr, index=X.index, columns=self.feature_names_)
   
    def _get_affected_columns(self, X: pd.DataFrame):
        affected_columns_, _ = super()._get_affected_columns(X)
        affected_columns_ = [col for col in affected_columns_ if X[col].value_counts().iloc[0] >= self.min_frequency]
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_
    
class CatLOOTransformer(CategoricalBasePreprocessor):
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
    def __init__(self, 
                 keep_original: bool = False,
                 sigma: float = 0.0, 
                 random_state: int | None = None,
                 **kwargs
                 ):
        super().__init__(keep_original=keep_original)
        self.sigma = sigma
        self.random_state = random_state
        self.__name__ = "CatLOOTransformer"

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series):
        X = X_in.copy()
        y = y_in.copy()

        if y is None:
            raise ValueError("Leave-One-Out encoding requires a target variable `y`.")

        # Fit Leave-One-Out encoder
        self.loo_ = LeaveOneOutEncoder(
            cols=list(X.columns),
            sigma=self.sigma,
            random_state=self.random_state
        )

        self.modes = X.mode().iloc[0]

        for col in X.columns:
            if X[col].isna().sum()>0:
                X[col] = X[col].fillna(self.modes[col])

        # TODO: Implement for multi-class
        if y.nunique()==2:
            self.loo_.fit(X, (y==y.iloc[0]).astype(float))
        else:
            self.loo_.fit(X, y.astype(float))

        return self

    def _transform(self, X_in: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X_out = X_in.copy()
        for col in X_out.columns:
            if X_out[col].isna().sum()>0:
                X_out[col] = X_out[col].fillna(self.modes[col])
        X_out = self.loo_.transform(X_out)
        X_out.columns = [i + '_LOO' for i in X_out.columns]

        return X_out
    
class TargetEncodingTransformer(CategoricalBasePreprocessor):
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
    def __init__(self, 
                 keep_original: bool = False,
                 sigma: float = 0.0, 
                 random_state: int | None = None,
                 **kwargs

                 ):
        super().__init__(keep_original=keep_original)
        self.sigma = sigma
        self.random_state = random_state

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series):
        X = X_in.copy()
        y = y_in.copy()

        if y is None:
            raise ValueError("Leave-One-Out encoding requires a target variable `y`.")

        # Fit Leave-One-Out encoder
        self.loo_ = TargetEncoder(
            # cols=list(X.columns),
            # sigma=self.sigma,
            # random_state=self.random_state
        )

        # TODO: Implement for multi-class
        if y.nunique()==2:
            self.loo_.fit(X, (y==y.iloc[0]).astype(float))
        else:
            self.loo_.fit(X, y.astype(float))

        return self

    def _transform(self, X_in: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X_out = X_in.copy()
        X_out = self.loo_.transform(X_out)
        X_out = pd.DataFrame(X_out, index=X_in.index, columns=X_in.columns)
        return X_out


class DropCatTransformer(CategoricalBasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(keep_original=False)
        self.drop_cols = []

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series):
        self.drop_cols = X_in.columns.tolist()
        if len(self.drop_cols) == 0:
            print("Warning: No categorical columns to drop.")

    def _transform(self, X_in: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(index=X_in.index)

    def _get_affected_columns(self, X: pd.DataFrame):
        affected_columns_, unaffected_columns_ = super()._get_affected_columns(X)
        if len(affected_columns_) == X.shape[1]:
            print("Warning: Dataset has only categorical columns. Don't attempt dropping them.")
            affected_columns_ = []
            unaffected_columns_ = X.columns.tolist()
        return affected_columns_, unaffected_columns_


class OOFTargetEncoder(CategoricalBasePreprocessor):
    """
    KFold out-of-fold target encoding (regression / binary / multiclass)
    Interpretation A:
      - fit(...) computes + stores OOF TRAIN encodings and full-train stats
      - transform(..., is_train=True) returns stored train encodings
      - transform(..., is_train=False) encodes new data using full stats
    """

    def __init__(self, 
                 target_type:str,
                 n_splits:int=5,
                 alpha:float=10.0,
                 random_state:int=42,
                 keep_original: bool = False,
                 ):
        super().__init__(keep_original=keep_original)
        assert target_type in {"regression","binary","multiclass"}
        self.target_type = target_type
        self.n_splits = n_splits
        self.alpha = alpha
        self.random_state = random_state

    # -----------------------------------------------------------
    def _fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        X = X.astype('object')
        self.cols_ = list(X.columns)

        # convert y to matrix by target_type
        if self.target_type == "regression":
            kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
            Y = y.values[:,None]

        elif self.target_type == "binary":
            kf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
            # require y {0,1} or {labelA,labelB}
            if y.dtype.name == "category":
                y = y.cat.codes
            classes = np.unique(y)
            assert len(classes) == 2, "binary target_type but >2 classes"
            Y = (y.values == classes[-1]).astype(float)[:,None]
            self.classes_ = classes

        else: # multiclass
            kf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
            if y.dtype.name == "category":
                y = y.cat.codes
            classes = np.unique(y)
            self.classes_ = classes
            K = len(classes)
            Y = np.zeros((len(y),K))
            for i,c in enumerate(classes):
                Y[:,i] = (y.values==c).astype(float)

        self.Y_ = Y
        self.X_ = X

        store = []

        for col in self.cols_:
            oof = np.zeros((len(X), Y.shape[1]))
            col_values = X[col]

            for tr,val in kf.split(X, y):
                df = pd.DataFrame({"cat":X[col]})
                for j in range(Y.shape[1]):
                    df[f"y{j}"] = Y[:,j]

                g = df.iloc[tr].groupby("cat", observed=True).agg(["count","mean"])
                for j in range(Y.shape[1]):
                    m = g[("y"+str(j),"mean")]
                    c = g[("y"+str(j),"count")]
                    enc = (m*c + self.alpha*m.mean())/(c + self.alpha)
                    oof[val,j] = col_values.iloc[val].map(enc).fillna(m.mean())

            # names
            if Y.shape[1]==1:
                names=[f"{col}__te"]
            else:
                names=[f"{col}__te_class{j}" for j in range(Y.shape[1])]
            store.append(pd.DataFrame(oof, columns=names, index=X_in.index))

        self.train_encoded_ = pd.concat(store, axis=1)

        # full train stats (for test/inference)
        full_stats = {}
        for col in self.cols_:
            df = pd.DataFrame({"cat":X[col]})
            for j in range(Y.shape[1]):
                df[f"y{j}"] = Y[:,j]
            g = df.groupby("cat", observed=True).agg(["count","mean"])
            full_stats[col] = g
        self.full_stats_ = full_stats

        return self

    # -----------------------------------------------------------
    def _transform(self, X_in, is_train:bool=False, **kwargs):
        X = pd.DataFrame(X_in).reset_index(drop=True)
        X = X.astype('object')

        if is_train:
            # return stored OOF train encodings
            # rather than recomputing
            assert hasattr(self,"train_encoded_"), "fit() not called"
            return self.train_encoded_.copy()

        # else new data: use full_stats
        out = []
        for col in self.cols_:
            arr = np.zeros((len(X), self.Y_.shape[1]))
            col_series = X[col]
            g = self.full_stats_[col]

            for j in range(self.Y_.shape[1]):
                m = g[("y"+str(j),"mean")]
                c = g[("y"+str(j),"count")]
                enc = (m*c + self.alpha*m.mean())/(c + self.alpha)
                arr[:,j] = col_series.map(enc).fillna(m.mean())

            if self.Y_.shape[1]==1:
                names=[f"{col}__te"]
            else:
                names=[f"{col}__te_class{j}" for j in range(self.Y_.shape[1])]
            out.append(pd.DataFrame(arr, columns=names, index=X_in.index))

        return pd.concat(out, axis=1)
