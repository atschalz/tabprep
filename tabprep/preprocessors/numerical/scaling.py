import pandas as pd
import numpy as np
from tabprep.preprocessors.base import NumericBasePreprocessor
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from skrub import SquashingScaler
from kditransform import KDITransformer

from typing import Literal

# TODO: Check whether a common base preprocessor for scalers makes sense
class StandardScalerPreprocessor(NumericBasePreprocessor):
    def __init__(self, 
                 keep_original: bool = False,
                 **kwargs
                #  scaler_kwargs: dict = dict(),
                 ):
        super().__init__(keep_original=keep_original)
        self.scaler = StandardScaler()

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            self.scaler.fit(X[self.affected_columns_])
        
    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            X_scaled = self.scaler.transform(X[self.affected_columns_])
            return pd.DataFrame(X_scaled, columns=[i+'_scaled' for i in X.columns], index=X.index)
        else:
            return pd.DataFrame()

class RobustScalerPreprocessor(NumericBasePreprocessor):
    def __init__(self, 
                 keep_original: bool = False,
                 **kwargs
                #  scaler_kwargs: dict = dict(),
                 ):
        super().__init__(keep_original=keep_original)
        self.scaler = RobustScaler()

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            self.scaler.fit(X[self.affected_columns_])
        
    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            X_scaled = self.scaler.transform(X[self.affected_columns_])
            return pd.DataFrame(X_scaled, columns=[i+'_quantile' for i in X.columns], index=X.index)
        else:
            return pd.DataFrame()

class QuantileScalerPreprocessor(NumericBasePreprocessor):
    def __init__(self, 
                 keep_original: bool = False,
                 scaler_kwargs: dict = dict(random_state=42, n_quantiles=1000, output_distribution='uniform'),
                 **kwargs
                 ):
        super().__init__(keep_original=keep_original)
        self.scaler = QuantileTransformer(**scaler_kwargs)

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            self.scaler.fit(X[self.affected_columns_])
        
    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            X_scaled = self.scaler.transform(X[self.affected_columns_])
            return pd.DataFrame(X_scaled, columns=[i+'_quantile' for i in X.columns], index=X.index)
        else:
            return pd.DataFrame()

class SquashingScalerPreprocessor(NumericBasePreprocessor):
    def __init__(self, keep_original: bool = False, **kwargs):
        super().__init__(keep_original=keep_original)
        self.scaler = SquashingScaler()

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            self.scaler.fit(X[self.affected_columns_])

        
    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            X_scaled = self.scaler.transform(X[self.affected_columns_])
            return pd.DataFrame(X_scaled, columns=[i+'_squashed' for i in X.columns], index=X.index)
        else:
            return pd.DataFrame()
        
class KDITransformerPreprocessor(NumericBasePreprocessor):
    # NOTE: Taken from TabPFNv2 codebase
    def __init__(self, 
                 keep_original: bool = False, 
                 alpha: float = 1.0,
                 output_distribution: Literal["uniform", "normal"] = "uniform",
                 fillna: Literal['mean', 'median'] = 'mean',
                 **kwargs
                 ):
        super().__init__(keep_original=keep_original)
        self.scaler = KDITransformer(alpha=alpha, output_distribution=output_distribution)
        self.fillna = fillna
        self.nan_means = {}

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()

        for col in X.columns:
            if self.fillna == 'mean':
                self.nan_means[col] = X[col].mean()
            elif self.fillna == 'median':
                self.nan_means[col] = X[col].median()
            else:
                raise ValueError(f'fillna must be "mean" or "median", got {self.fillna}')
            if X[col].isna().any():
                X[col] = X[col].fillna(self.nan_means[col])

        if len(self.affected_columns_) > 0:
            self.scaler.fit(X[self.affected_columns_])

        
    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            nan_mask = np.isnan(X)          
            for col in X.columns:
                if X[col].isna().any():
                    X[col] = X[col].fillna(self.nan_means[col])
            
            X_scaled = self.scaler.transform(X[self.affected_columns_])
            X_scaled[nan_mask] = np.nan
            return pd.DataFrame(X_scaled, columns=[i+'_kdi' for i in X.columns], index=X.index)
        else:
            return pd.DataFrame()
