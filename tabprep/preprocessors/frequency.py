import pandas as pd
from tabprep.preprocessors.base import BasePreprocessor

class FrequencyEncoder(BasePreprocessor):
    def __init__(self, 
                 keep_original: bool = True,
                 only_categorical: bool = True,
                 candidate_cols: list = None, 
                 use_filters: bool = True,
                 **kwargs
                 ):
        super().__init__(keep_original=keep_original)
        self.only_categorical = only_categorical
        self.candidate_cols = candidate_cols
        self.use_filters = use_filters
        
        self.freq_maps = {}

    # TODO: Move to detector
    def filter_candidates_by_distinctiveness(self, X: pd.DataFrame) -> list:
        candidate_cols = []
        for col in X.columns:
            x_new = X[col].map(X[col].value_counts().to_dict())
            if all((pd.crosstab(X[col],x_new)>0).sum()==1):
                continue
            else:
                candidate_cols.append(col)

        return candidate_cols

    def _fit(self, X_in: pd.DataFrame, y_in: pd.Series = None):
        X = X_in.copy()

        for col in X.columns:
            x = X[col]
            self.freq_maps[x.name] = x.value_counts().to_dict()

        return self
    
    def _transform(self, X_in):
        X = X_in.copy()

        new_cols = []
        for col in X.columns:
            x = X[col]
            if x.name in self.freq_maps:
                new_col = x.map(self.freq_maps[x.name]).astype(float).fillna(0)
                new_col.name = x.name + "_freq"
                new_cols.append(new_col)
            else:
                continue

        return pd.concat(new_cols, axis=1)
    
    def _get_affected_columns(self, X: pd.DataFrame):
        if self.candidate_cols is None:
            if self.only_categorical:
                affected_columns_ = X.select_dtypes(include=['category', 'object']).columns.tolist()
            else:
                affected_columns_ = X.columns.tolist()
            if self.use_filters:
                affected_columns_ = self.filter_candidates_by_distinctiveness(X[affected_columns_])
        else:
            affected_columns_ = self.candidate_cols
        
        unaffected_columns_ = X.drop(columns=affected_columns_).columns.tolist()

        return affected_columns_, unaffected_columns_