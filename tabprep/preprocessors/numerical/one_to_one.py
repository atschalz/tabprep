import pandas as pd
import numpy as np
from tabprep.preprocessors.base import NumericBasePreprocessor
from sklearn.preprocessing import SplineTransformer, OneHotEncoder

from typing import Literal, List

class TrigonometricTransformer(NumericBasePreprocessor):
    # NOTE: Taken from TabPFNv2 codebase
    def __init__(
        self, 
        keep_original: bool = True, 
        operations: List[Literal['sin', 'cos', 'tan', 'sincos']] = ['sin', 'cos', 'tan', 'sincos']
    ):
        super().__init__(keep_original=keep_original)
        self.operations = operations

    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
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