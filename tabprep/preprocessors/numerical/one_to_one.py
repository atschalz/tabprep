import pandas as pd
import numpy as np
from tabprep.preprocessors.base import NumericBasePreprocessor
from sklearn.preprocessing import SplineTransformer, OneHotEncoder

from typing import Literal, List

from sklearn.base import clone
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor
from tabprep.preprocessors.base import NumericBasePreprocessor
from tabprep.utils.modeling_utils import make_cv_function, adjust_target_format

class TrigonometricTransformer(NumericBasePreprocessor):
    # NOTE: Taken from TabPFNv2 codebase
    def __init__(
        self, 
        keep_original: bool = True, 
        operations: List[Literal['sin', 'cos', 'tan', 'sincos']] = ['sin', 'cos', 'tan', 'sincos'],
        **kwargs
    ):
        super().__init__(keep_original=keep_original)
        self.operations = operations

    def _transform(self, X_in: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X = X_in.copy()
        X_transformed = pd.DataFrame(index=X.index)
        # FIXME: Can simply apply operations on whole DataFrame
        for col in self.affected_columns_:
            for op in self.operations:
                if op == 'sin':
                    X_transformed[f"{col}_sin"] = np.sin(X[col])
                elif op == 'cos':
                    X_transformed[f"{col}_cos"] = np.cos(X[col])
                elif op == 'tan':
                    X_transformed[f"{col}_tan"] = np.tan(X[col])
                elif op == 'sincos':
                    X_transformed[f"{col}_sincos"] = np.sin(X[col]) * np.cos(X[col])
                else:
                    raise ValueError(f"Unsupported operation '{op}'. Supported operations are 'sin', 'cos', 'tan', 'sincos'.")
        return X_transformed
    
class OptimalBinner(NumericBasePreprocessor):
    def __init__(self, 
                 target_type,
                 candidate_bins=(2, 4, 8, 16, 32, 64, 128, 256, 512,1024,2048),
                 strategy="uniform",
                 encode="ordinal",
                 **kwargs
                 ):
        """
        Parameters
        ----------
        candidate_bins : iterable of int
            Possible numbers of bins to try for each feature.
        strategy : str
            KBinsDiscretizer strategy ("uniform", "quantile", "kmeans").
        encode : str
            Encoding type for KBinsDiscretizer.
        """
        super().__init__(keep_original=False)
        self.target_type = target_type
        self.candidate_bins = tuple(candidate_bins)
        self.strategy = strategy
        self.encode = encode
        
        self.multi_as_bin = False
        if target_type == 'binary':
            self.model = TargetMeanClassifier()
        elif target_type == 'multiclass':
            self.multi_as_bin = True          
            self.model = TargetMeanClassifier()
        else:
            self.model = TargetMeanRegressor()
        self.cv = make_cv_function(target_type) 
        
    def _fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        if self.multi_as_bin:
            y = adjust_target_format(y, target_type=self.target_type, multi_as_bin=True)
        X = pd.DataFrame(X).copy()
        self.n_bin_per_feature_ = {}
        self.col_binners_ = {}
        
        for col in X.columns:
            x_col = X[[col]].copy()
            x_col = x_col.dropna()
            # TODO: Fix many-nan setting
            y_col = y[x_col.index]
            if x_col.shape[0] < 10 or y_col.value_counts().min() < 5: # FIXME Make folds a parameter instead of using 5
                continue
            # start with no binning (baseline)
            est = Pipeline([
                # ('fillna', SimpleImputer(strategy='mean')),
                # ('bin', KBinsDiscretizer(n_bins=n_bins, strategy=self.strategy, encode=self.encode)),
                ('model', clone(self.model))
            ])
            
            best_score = np.mean(self.cv(x_col, y_col, est)['scores'])

            best_bin = None
            
            for n_bins in sorted(self.candidate_bins, reverse=True):
                if n_bins >= x_col[col].nunique():
                    continue
                
                est = Pipeline([
                    ('fillna', SimpleImputer(strategy='mean')),
                    ('bin', KBinsDiscretizer(n_bins=n_bins, strategy=self.strategy, encode=self.encode)),
                    ('model', clone(self.model))
                ])
                
                scores = self.cv(x_col, y_col, est)['scores']
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_bin = n_bins
            
            if best_bin is not None:
                self.n_bin_per_feature_[col] = best_bin
                binner = KBinsDiscretizer(n_bins=best_bin,
                                          strategy=self.strategy,
                                          encode=self.encode)
                binner.fit(x_col)
                self.col_binners_[col] = binner
        
        return self
    
    def _transform(self, X_in, **kwargs):
        X_out = X_in.copy()
        
        for col, binner in self.col_binners_.items():
            mask = X_in[col].notna().values
            X_out.loc[mask,col] = binner.transform(X_out.loc[mask,[col]]).ravel()
        
        return X_out


class NearestNeighborDistanceTransformer(NumericBasePreprocessor):
    def __init__(self, keep_original=True, min_cardinality=6, **kwargs):
        super().__init__(keep_original=keep_original)
        self.min_cardinality = min_cardinality
        
    def _fit(self, X_in, y=None):
        self.unique_values_per_col = {
            col: np.sort(X_in[col].dropna().unique())
            for col in X_in.columns
        }
        return self
    
    def _transform(self, X_in, **kwargs):
        X_out = pd.DataFrame(index=X_in.index)
        for col in X_in.columns:
            if col not in self.unique_values_per_col:
                continue  # column not fitted

            uniques = self.unique_values_per_col[col]

            # compute nearest neighbor distance for each value
            col_values = X_in[col].values.reshape(-1, 1)
            diffs = np.abs(col_values - uniques.reshape(1, -1))
            nearest_dist = diffs.min(axis=1)

            X_out[f"{col}_nn_dist"] = nearest_dist
        
        return X_out
    
    def _get_affected_columns(self, X: pd.DataFrame):
        affected_columns_ = super()._get_affected_columns(X)[0]
        affected_columns_ = [col for col in affected_columns_ if X[col].nunique() > self.min_cardinality]
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_
    
    def fit_transform(self, X_in: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X_in.copy()
        self.fit(X, y)
        X = X[self.affected_columns_]
        Xt = pd.DataFrame(index=X_in.index)

        for col in X.columns:
            uniques = self.unique_values_per_col[col]

            # Compute distances for each unique
            diffs = np.diff(uniques)
            left_dists = np.r_[np.inf, diffs]    # distance to previous
            right_dists = np.r_[diffs, np.inf]   # distance to next
            nearest = np.minimum(left_dists, right_dists)

            # Map value → nearest distance
            dist_map = dict(zip(uniques, nearest))

            # Vectorized assignment
            Xt[f"{col}_nn_dist"] = X_in[col].map(dist_map)

        if self.keep_original:
            return pd.concat([X_in[self.unaffected_columns_], X_in[self.affected_columns_], Xt], axis=1)
        else:
            return pd.concat([X_in[self.unaffected_columns_], Xt], axis=1)
