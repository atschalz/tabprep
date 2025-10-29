import pandas as pd
import numpy as np
from tabprep.preprocessors.base import NumericBasePreprocessor
from sklearn.preprocessing import SplineTransformer, OneHotEncoder

from typing import Literal

class SplinePreprocessor(NumericBasePreprocessor):
    # NOTE: Taken from TabPFNv2 codebase
    def __init__(self, 
                 keep_original: bool = True, 
                 fillna: Literal['mean', 'median'] = 'median',
                 min_cardinality = 10,
                 transformer_kwargs: dict = dict(
                    #  include_bias=False,
                    #  n_knots=5,
                    #  degree=3,
                    #  knots = 'uniform',
                    #  extrapolation = 'constant',
                     ),
                **kwargs
                 ):
        super().__init__(keep_original=keep_original)
        self.transformer = SplineTransformer(include_bias=False, **transformer_kwargs)
        self.fillna = fillna
        self.min_cardinality = min_cardinality
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
            self.transformer.fit(X[self.affected_columns_])

        
    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        # TODO: nan logic reappers, consider general class
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            nan_mask = np.isnan(X)          
            for col in X.columns:
                if X[col].isna().any():
                    X[col] = X[col].fillna(self.nan_means[col])
            
            X_scaled = self.transformer.transform(X[self.affected_columns_])

            return pd.DataFrame(X_scaled, columns=self.transformer.get_feature_names_out(X.columns), index=X.index)
        else:
            return pd.DataFrame()

    def _get_affected_columns(self, X: pd.DataFrame):
        affected_columns_ = X.select_dtypes(include=['number']).columns.tolist()
        affected_columns_ = [col for col in affected_columns_ if X[col].nunique() > self.min_cardinality]  
        
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_

# TODO: Rather align with the categorical OneHotEncoder and maybe make this a general preprocessor, while the cat and num preprocessors subclass the general one
class NumericalOneHotPreprocessor(NumericBasePreprocessor):
    """
    One-hot encode categorical columns with one level dropped per feature (to avoid multicollinearity).
    - Column naming: <original_col>__<category_value>
    - Unknown categories at transform time are ignored (all-zeros for that feature).
    """

    def __init__(
            self, 
            keep_original: bool = False,
            drop: Literal['if_binary', 'first'] = 'if_binary',
            min_frequency: int = 20,
            **kwargs
            ):
        super().__init__(keep_original=keep_original)
        self.drop = drop
        self.min_frequency = min_frequency

    def _fit(self, X: pd.DataFrame, y=None):
        # categories = [sorted(pd.Series(col).unique().tolist()) for _, col in X.items()]
        self.encoder_ = OneHotEncoder(
            # categories=categories,
            drop=self.drop,
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=self.min_frequency
        ).fit(X)
        self.feature_names_ = self.encoder_.get_feature_names_out(X.columns)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.encoder_.transform(X)
        return pd.DataFrame(arr, index=X.index, columns=self.feature_names_)

    def _get_affected_columns(self, X: pd.DataFrame):
        affected_columns_, _ = super()._get_affected_columns(X)
        affected_columns_ = [col for col in affected_columns_ if X[col].value_counts().iloc[0] >= self.min_frequency]
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_
    
class LowCardinalityOneHotPreprocessor(NumericalOneHotPreprocessor):
    def __init__(
            self,
            max_cardinality: int = 10,
            **kwargs
            ):
        kwargs['min_frequency'] = 1
        super().__init__(**kwargs)
        self.max_cardinality = max_cardinality

    def _get_affected_columns(self, X: pd.DataFrame):
        affected_columns_ = [col for col in X.columns if X[col].nunique() <= self.max_cardinality]
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_
    