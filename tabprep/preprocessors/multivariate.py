import pandas as pd
import numpy as np
from tabprep.preprocessors.base import BasePreprocessor, NumericBasePreprocessor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import FastICA, TruncatedSVD, PCA, KernelPCA, SparsePCA, DictionaryLearning, FactorAnalysis, LatentDirichletAllocation, NMF
from sklearn.pipeline import Pipeline

from typing import Literal, Optional, Dict, Any
from collections import Counter
from category_encoders import LeaveOneOutEncoder

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from inspect import signature

# For LinearFeatureAdder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import OneHotEncoder

from sklearn.kernel_approximation import RBFSampler


class SKLearnDecompositionPreprocessor(BasePreprocessor):
    # NOTE: Taken from TabPFNv2 codebase
    def __init__(self, 
                 transformer_cls: type,
                 transformer_init_kwargs: dict = dict(),
                 scale_X: bool = True,
                 only_numerical: bool = False,
                 non_negative: bool = False,
                 keep_original: bool = True,
                 n_components: Optional[int] = None,
                 **kwargs
                 ):
        super().__init__(keep_original=keep_original)
        self.scale_X = scale_X
        self.only_numerical = only_numerical
        self.non_negative = non_negative
        self.n_components = n_components
        # FIXME: Rather use identity preprocessor
        if scale_X:
            self.scaler = StandardScaler()
        self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.nan_filler = SimpleImputer(strategy='mean')
        self.transformer_cls = transformer_cls
        self.transformer_init_kwargs = transformer_init_kwargs
        self.new_col_prefix = ''

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()
        
        self.initialize_transformer(X)
        
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        X[self.cat_cols] = self.cat_encoder.fit_transform(X[self.cat_cols])

        if self.scale_X:
            X = self.scaler.fit_transform(X)

        X = self.nan_filler.fit_transform(X)

        self.transformer.fit(X)
        return self
        
    def _transform(self, X_in: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # TODO: nan logic reappers, consider general class
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            X = X_in.copy()
            X[self.cat_cols] = self.cat_encoder.transform(X[self.cat_cols])
            if self.scale_X:
                X = self.scaler.transform(X)
            X = self.nan_filler.transform(X)
            
            # Make sure that test data is also non-negative if required (e.g. for NMF, LDA)
            if self.non_negative:
                X[X < 0] = 0

            X_out = pd.DataFrame(self.transformer.transform(X), index=X_in.index)
            X_out.columns = [f"{self.new_col_prefix}_{i}" for i in range(X_out.shape[1])]
            return X_out
        else:
            return pd.DataFrame()

    def _get_affected_columns(self, X: pd.DataFrame):
        """Return (affected_columns_, unaffected_columns_). Subclasses decide the rule."""
        if self.only_numerical:
            affected_columns_ = X.select_dtypes(include=['number']).columns.tolist()
        else:
            affected_columns_ = X.columns.tolist()
        
        if self.non_negative:
            affected_columns_ = [col for col in affected_columns_ if X[col].min() >= 0]
            if len(affected_columns_) < 2:
                affected_columns_ = []  # LDA needs at least 2 non-negative features
        
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()

        return affected_columns_, unaffected_columns_

    def initialize_transformer(self, X: pd.DataFrame) -> int:
        if self.n_components is None:
            self.n_components = max(
                1,
                min(
                    int(X.shape[0] * 0.8) // 10 + 1,
                    X.shape[1] // 2,
                ),
            )

        self.transformer = self.transformer_cls(
            n_components=self.n_components,
            **self.transformer_init_kwargs
        )


class FastICAPreprocessor(SKLearnDecompositionPreprocessor):
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 **kwargs
                 ):
        super().__init__(
            keep_original=keep_original,
            transformer_cls=FastICA,
            transformer_init_kwargs=dict(random_state=random_state),
            **kwargs
            )
        self.new_col_prefix = 'ica'

class SVDPreprocessor(SKLearnDecompositionPreprocessor):
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 **kwargs
                 ):
        super().__init__(
            keep_original=keep_original,
            transformer_cls=TruncatedSVD,
            transformer_init_kwargs=dict(algorithm="arpack", random_state=random_state),
            **kwargs
            )
        self.new_col_prefix = 'svd'

class PCAPreprocessor(SKLearnDecompositionPreprocessor):
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 **kwargs
                 ):
        # TODO: Consider using IncrementalPCA for large datasets
        super().__init__(
            keep_original=keep_original,
            transformer_cls=PCA,
            transformer_init_kwargs=dict(random_state=random_state),
            **kwargs
            )
        self.new_col_prefix = 'pca'

class KernelPCAPreprocessor(SKLearnDecompositionPreprocessor):
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'] = 'rbf',
                 kernel_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs
                 ):
        super().__init__(
            keep_original=keep_original,
            transformer_cls=KernelPCA,
            transformer_init_kwargs=dict(
                kernel=kernel,
                random_state=random_state, 
                n_jobs=-1,
                **(kernel_kwargs if kernel_kwargs is not None else dict()),
            ),
            **kwargs
        )
        self.new_col_prefix = f'{kernel}-PCA'

class SparsePCAPreprocessor(SKLearnDecompositionPreprocessor):
    ''' SparsePCA - Aims for extracting sparse components instead of dense components as standard PCA does.'''
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 **kwargs
                 ):
        super().__init__(
            keep_original=keep_original,
            transformer_cls=SparsePCA,
            transformer_init_kwargs=dict(random_state=random_state, n_jobs=-1),
            **kwargs
            )
        self.new_col_prefix = 'sparsepca'

class DictionaryLearningPreprocessor(SKLearnDecompositionPreprocessor):
    ''' DictionaryLearning - Sparse encoding of dense data.'''
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 **kwargs
                 ):
        super().__init__(
            keep_original=keep_original,
            transformer_cls=DictionaryLearning,
            transformer_init_kwargs=dict(random_state=random_state, n_jobs=-1, transform_algorithm='lasso_lars'),
            **kwargs
            )
        self.new_col_prefix = 'dictlearn'

class FactorAnalysisPreprocessor(SKLearnDecompositionPreprocessor):
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 **kwargs
                 ):
        super().__init__(
            keep_original=keep_original,
            transformer_cls=FactorAnalysis,
            transformer_init_kwargs=dict(random_state=random_state),
            **kwargs
            )
        self.new_col_prefix = 'factoranalysis'


class LDAPreprocessor(SKLearnDecompositionPreprocessor):
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 **kwargs
                 ):
        super().__init__(
            keep_original=keep_original,
            transformer_cls=LatentDirichletAllocation,
            transformer_init_kwargs=dict(random_state=random_state, learning_method='batch', n_jobs=-1),
            only_numerical=True,
            non_negative=True,
            scale_X=False  # LDA requires non-negative inputs
            )
        self.new_col_prefix = 'lda'

class NMFPreprocessor(SKLearnDecompositionPreprocessor):
    def __init__(self, 
                 keep_original: bool = True,
                 random_state: int = 42,
                 **kwargs
                 ):
        super().__init__(
            keep_original=keep_original,
            transformer_cls=NMF,
            transformer_init_kwargs=dict(random_state=random_state, init='nndsvda', max_iter=500),
            non_negative=True,
            only_numerical=True,
            scale_X=False  # NMF requires non-negative inputs
            )
        self.new_col_prefix = 'nmf'
    
class DuplicateCountAdder(NumericBasePreprocessor):
    """
    A transformer that counts duplicate samples in the training set
    and appends a new feature with those counts at transform time.
    """

    def __init__(
            self, 
            feature_name: str = "duplicate_count", 
            cap: bool = False,
            **kwargs
            ):
        super().__init__(keep_original=True)
        self.feature_name = feature_name
        self.cap = cap

    def _fit(self, X_in: pd.DataFrame, y: pd.Series = None):
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

    def _transform(self, X_in: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

        X_out = pd.DataFrame(new_feat)
        return X_out

class DuplicateSampleLOOEncoder(BasePreprocessor):
    def __init__(
        self, new_name='LOO_duplicates',
        **kwargs
    ):
        super().__init__(keep_original=True)
        self.new_col = new_name

    def _fit(self, X, y):
        X_str = X.astype(str).sum(axis=1)

        self.u_id_map = {i:j for i,j in zip(X_str.unique(),list(range(X_str.nunique())))}

        X_uid = pd.DataFrame(X_str.map(self.u_id_map).astype(str))

        self.loo = LeaveOneOutEncoder()
        if y.nunique()==2:
            self.loo.fit(X_uid, (y==y.iloc[0]).astype(float))
        else:
            self.loo.fit(X_uid, y.astype(float))

        return self

    def _transform(self, X, **kwargs):
        X_out = X.copy()
        X_str = X.astype(str).sum(axis=1)
        X_uid = pd.DataFrame(X_str.map(self.u_id_map).astype(str))
        X_out = pd.DataFrame(self.loo.transform(X_uid), index=X.index) 
        X_out.columns = [self.new_col]
        return X_out
    
    def _get_affected_columns(self, X: pd.DataFrame):
        affected_columns_ = X.columns.tolist()
        unaffected_columns_ = []
        return affected_columns_, unaffected_columns_

@dataclass
class _CatSpec:
    frequent_values: set

# TODO: This function is vibe-coded, get rid of unnecessary stuff
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
        random_state: Optional[int] = 42,
        **kwargs
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
        self.__name__ = "LinearFeatureAdder"

        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # TODO: Extend to Ridge/Lasso/ElasticNet
        if self.linear_model_type == 'default':
            if self.target_type == "regression":
                self.model_ = LinearRegression(**self.model_params)
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
            enc = self.ohe_.transform(mapped.astype(str))
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
            enc = self.ohe_.transform(mapped.astype(str))
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
            self.ohe_ = self._make_ohe().fit(mapped.astype(str))
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

    def transform(self, X, **kwargs):
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

import numpy as np

class RandomFourierFeatureTransformer(NumericBasePreprocessor):
    def __init__(self, n_features=100, gamma=1.0, random_state=None, **kwargs):
        """
        n_features: number of Fourier features to generate
        gamma: RBF kernel parameter (1 / (2 * sigma^2))
        """
        super().__init__(keep_original=True)
        self.n_features = n_features
        self.gamma = gamma
        self.random_state = np.random.RandomState(random_state)
    
    def _fit(self, X_in, x_in=None):
        X = X_in.copy()
        n_input_features = X.shape[1]
        # Sample random projection matrix W ~ N(0, 2*gamma I)
        self.W = self.random_state.normal(
            loc=0, scale=np.sqrt(2 * self.gamma), 
            size=(n_input_features, self.n_features)
        )
        # Random phase b ~ Uniform(0, 2π)
        self.b = self.random_state.uniform(0, 2*np.pi, size=self.n_features)
        return self
    
    def _transform(self, X_in, **kwargs):
        X = X_in.copy()
        projection = X @ self.W + self.b
        X_out = np.sqrt(2.0 / self.n_features) * np.cos(projection)
        X_out.columns = [f'rff_{i}' for i in range(self.n_features)]
        return X_out

class SklearnRandomFourierFeatureTransformer(NumericBasePreprocessor):
    def __init__(self, n_features=100, gamma=1.0, random_state=None, **kwargs):
        """
        n_features: number of Fourier features to generate
        gamma: RBF kernel parameter (1 / (2 * sigma^2))
        """
        super().__init__(keep_original=True)
        self.n_features = n_features
        self.gamma = gamma
        self.random_state = random_state
    
    def _fit(self, X_in, x_in=None):
        X = X_in.copy()
        self.rff_ = RBFSampler(n_components=self.n_features, gamma=self.gamma, random_state=self.random_state)
        # TODO: Handle NaNs as with other numerical preprocessors
        self.rff_.fit(X.fillna(0))
        return self
    
    def _transform(self, X_in, **kwargs):
        X = X_in.copy()
        X_out = pd.DataFrame(self.rff_.transform(X.fillna(0)), index=X.index)
        X_out.columns = [f'rff_{i}' for i in range(self.n_features)]
        return X_out


