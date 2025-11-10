from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BasePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, keep_original=False, **kwargs): # TODO: Decide what to use as default
        """
        Parameters
        ----------
        keep_original : bool, default=False
            If True, keep original affected features alongside transformed ones.
        """
        self.keep_original = keep_original
        self.__name__ = self.__class__.__name__

    def fit(self, X: pd.DataFrame, y=None):
        self.affected_columns_, self.unaffected_columns_ = self._get_affected_columns(X)
        # NOTE: It is assumed that only affected columns are passed to _fit, so the subclass preprocessors fail if thats not the case
        if len(self.affected_columns_) > 0:
            self._fit(X[self.affected_columns_], y)
        else:
            print(f"[Warning] No affected columns found for {self.__name__}. Skipping fit.")

        return self

    def transform(self, X: pd.DataFrame, is_train=False) -> pd.DataFrame:
        if len(self.affected_columns_) == 0:
            return X.copy()
        
        Xt = self._transform(X[self.affected_columns_], is_train=is_train)
        # NOTE: if self.keep_original, the new names must be distinct and in 1-to-many the subclass must implement logic for appropriate naming itself

        if self.keep_original:
            # Ensure no column name clashes
            for col in self.unaffected_columns_:
                if col in Xt.columns:
                    raise ValueError(f"Column name clash: {col} exists in both unaffected and transformed dataframes.")
            return pd.concat([X[self.unaffected_columns_], X[self.affected_columns_], Xt], axis=1)
        else:
            return pd.concat([X[self.unaffected_columns_], Xt], axis=1)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, is_train=True)

    # ---- To be implemented by subclasses ----
    def _get_affected_columns(self, X: pd.DataFrame):
        """Return (affected_columns_, unaffected_columns_). Subclasses decide the rule."""
        return X.columns.tolist(), []

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series | None = None) -> pd.DataFrame:
        """Subclasses implement actual transformation."""
        pass

    def _transform(self, X: pd.DataFrame, is_train=False) -> pd.DataFrame:
        """Subclasses implement actual transformation."""
        return X

# TODO: If nothing else changes with these BasePreprocessors, move the _get_affected_columns to Base and delete them
class NumericBasePreprocessor(BasePreprocessor):
    def _get_affected_columns(self, X: pd.DataFrame, **kwargs):
        affected_columns_ = X.select_dtypes(include=['number']).columns.tolist()
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_

class CategoricalBasePreprocessor(BasePreprocessor):
    def _get_affected_columns(self, X: pd.DataFrame, **kwargs):
        affected_columns_ = X.select_dtypes(include=['category', 'object']).columns.tolist()
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_
    
# class NumericBasePreprocessorWithPreprocessing(BasePreprocessor):
# '''AUTO COMPLETED, TODO'''
#     def __init__(self, 
#                  keep_original=False, 
#                  preprocessing: Literal['none', 'minmax', 'standard'] = 'none'
#                  ):
#         super().__init__(keep_original=keep_original)
#         if preprocessing == 'none':
#             self.preprocessing = None
#         elif preprocessing == 'minmax':
#             from sklearn.preprocessing import MinMaxScaler
#             self.preprocessing = MinMaxScaler()
#         elif preprocessing == 'standard':
#             from sklearn.preprocessing import StandardScaler
#             self.preprocessing = StandardScaler()
#         else:
#             raise ValueError(f'preprocessing must be "none", "minmax" or "standard", got {preprocessing}')

    # def _get_affected_columns(self, X: pd.DataFrame):
    #     affected_columns_ = X.select_dtypes(include=['number']).columns.tolist()
    #     unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
    #     return affected_columns_, unaffected_columns_
