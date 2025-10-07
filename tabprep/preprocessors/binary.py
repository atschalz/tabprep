import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from tabprep.preprocessors.base import BasePreprocessor

from typing import List, Dict, Literal
from tabprep.utils.modeling_utils import adjust_target_format

class BaseBinaryTransformer(BasePreprocessor):
    """
    A base class for binary transformers.
    """
    def __init__(self, 
                 keep_original: bool = False, 
                 ):
        super().__init__(keep_original=keep_original)

    def _get_affected_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        affected_columns_ = X.columns[X.nunique() == 2].tolist()
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_
    

class BinaryToNumericTransformer(BaseBinaryTransformer):
    """
    A base class for binary transformers.
    """
    def __init__(self):
        super().__init__(keep_original=False)

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()
        self.positive_values_ = {}
        for col in X_in.columns:
            self.positive_values_[col] = X[col].value_counts().index[0]
        return self

    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.copy()
        X_transformed = pd.DataFrame(index=X.index)
        for col in self.affected_columns_:
            X_transformed[col] = (X[col] == self.positive_values_[col]).astype(int)
        return X_transformed

class BinarySumPreprocessor(BaseBinaryTransformer):
    """
    A Transformer that sums all binary columns into a single column.
    """
    def __init__(self, 
                 keep_original: bool = False, 
                 new_column_name: str = 'binary_sum'
                 ):
        super().__init__(keep_original=keep_original)
        self.new_column_name = new_column_name

    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.copy()
        if len(self.affected_columns_) > 0:
            X_transformed = pd.DataFrame(index=X.index)
            X_transformed[self.new_column_name] = X[self.affected_columns_].sum(axis=1)
            return X_transformed
        else:
            return pd.DataFrame()

    def _get_affected_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        affected_columns_ = super()._get_affected_columns(X)[0]
        # Sum only makes sense if at least two numeric binary columns are present
        affected_columns_ = [col for col in affected_columns_ if set(X[col].dropna().unique()).issubset({0, 1})]
        if len(affected_columns_) < 3:
            affected_columns_ = []
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_

class BinarySumByTargetDirectionPreprocessor(BaseBinaryTransformer):
    """
    A Transformer that sums all binary columns into a single column, grouped by the target variable.
    """
    # TODO: Make it subclass the binary_sum preprocessor
    def __init__(self, 
                 target_type: Literal['binary', 'multiclass', 'regression'] = 'binary',
                 new_column_name: str = 'binary_sum_by_target'
                 ):
        super().__init__(keep_original=True)
        self.new_column_name = new_column_name
        self.target_means_ = {}

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()
        y = y_in.copy()  # Just to avoid linter warnings about unused variable
        self.ordinal_encoder_ = OrdinalEncoder(dtype=int)
        
        X = pd.DataFrame(self.ordinal_encoder_.fit_transform(X[self.affected_columns_]), columns=self.affected_columns_, index=X.index)
        y = adjust_target_format(y, target_type, multi_as_bin=True)

        if y_in is None:
            raise ValueError("y_in must be provided for BinarySumByTargetPreprocessor.")
        for col in self.affected_columns_:
            self.target_means_[col] = X.groupby(y_in)[col].mean().sort_values().to_dict()
        return self

    def _transform(self, X_in: pd.DataFrame) -> pd.DataFrame:
        X = X_in.copy()
        X = pd.DataFrame(self.ordinal_encoder_.fit_transform(X[self.affected_columns_]), columns=self.affected_columns_, index=X.index)

        if len(self.affected_columns_) > 0:
            X_transformed = pd.DataFrame(index=X.index)
            X_transformed[self.new_column_name] = 0
            for col in self.affected_columns_:
                if col not in self.target_means_:
                    raise ValueError(f"Column '{col}' was not seen during fit.")
                sorted_values = sorted(self.target_means_[col], key=self.target_means_[col].get)
                mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
                X_transformed[self.new_column_name] += X[col].map(mapping).fillna(0)
            return X_transformed


from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from tabprep.utils.misc import cat_as_num
class BinaryJaccardGrouper(BaseBinaryTransformer):
    def __init__(self, n_clusters=None, threshold=0.3):
        """
        n_clusters : int, optional
            Force a fixed number of groups. If None, use threshold.
        threshold : float
            Max Jaccard distance within a cluster (ignored if n_clusters set).
        keep_original : bool
            If True, append counts to original features instead of replacing.
        """
        super().__init__(keep_original=True)
        self.n_clusters = n_clusters
        self.threshold = threshold

    def _fit(self, X_in, y_in=None):
        X = X_in.copy()
        X = cat_as_num(X)

        # compute Jaccard clustering only if we have at least 2 binary features
        dist = pdist(X.T.values, metric="jaccard")
        Z = linkage(dist, method="average")
        if self.n_clusters is not None:
            self.labels_ = fcluster(Z, self.n_clusters, criterion="maxclust")
        else:
            self.labels_ = fcluster(Z, self.threshold, criterion="distance")

        return self

    def _transform(self, X_in: pd.DataFrame):
        X = X_in.copy()
        X = cat_as_num(X)

        out = pd.DataFrame(index=X.index)

        # group binary columns
        groups = {}
        for g in np.unique(self.labels_):
            cols = [c for c, lab in zip(X.columns, self.labels_) if lab == g]
            groups[g] = cols

        for g, cols in groups.items():
            out[f"group{g}_count"] = X[cols].sum(axis=1)

        return out

    def _get_affected_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        affected_columns_ = super()._get_affected_columns(X)[0]
        # Sum only makes sense if at least two numeric binary columns are present
        affected_columns_ = [col for col in affected_columns_ if set(X[col].dropna().unique()).issubset({0, 1})]
        if len(affected_columns_) < 3:
            affected_columns_ = []
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()
        return affected_columns_, unaffected_columns_