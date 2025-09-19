import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from tabprep.preprocessors.base import BasePreprocessor, CategoricalBasePreprocessor

from typing import List, Dict

class ToCategoricalTransformer(BasePreprocessor):
    """
    A Transformer that converts all columns to categorical dtype,
    remembering the categories observed during fit.
    """
    def __init__(
            self, 
            keep_original: bool = False, 
            only_numerical: bool = False,
            min_cardinality: int = 6
            ):
        super().__init__(keep_original=keep_original)
        self.min_cardinality = min_cardinality
        self.only_numerical = only_numerical
        self.categories_ = {}

    def _fit(self, X_in, y_in=None):
        X = X_in.copy()

        # Store categories for each column
        # TODO: Might need to do something with values unseen during fit (& also NAs)
        self.categories_ = {
            col: pd.Series(X[col], dtype="category").cat.categories for col in X.columns
        }
        return self

    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X_out = X_in.copy()

        for col in self.categories_:
            if col not in self.categories_:
                raise ValueError(f"Column '{col}' was not seen during fit.")

            cat_col = pd.Categorical(
                X_out[col],
                categories=self.categories_[col]
            )

            X_out[f"{col}_cat"] = cat_col

        return X_out

    def _get_affected_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        if self.only_numerical:
            affected_columns_ = X.select_dtypes(include=['number']).columns.tolist()
        else:
            affected_columns_ = X.columns.tolist()
        
        affected_columns_ = [col for col in affected_columns_ if X[col].nunique() >= self.min_cardinality]
        
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_

class CatAsNumTransformer(CategoricalBasePreprocessor):
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
        keep_original=False,
        handle_unknown: str = "use_encoded_value",
        unknown_value: float | int = -1,
        dtype=np.float64,
    ):
        super().__init__(keep_original=keep_original)
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.dtype = dtype

    def _fit(self, X_in: pd.DataFrame, y_in=None):
        X = X_in.copy()
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

    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X_out = X_in.copy()

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

        if self.keep_original:
            X_out.columns = [f"{col}_num" for col in X_out.columns]

        return X_out
    

