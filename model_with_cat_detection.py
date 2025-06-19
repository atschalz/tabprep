from tabrepo.benchmark.models.ag import RealMLPModel
from ft_detection import FeatureTypeDetector
import pandas as pd
import numpy as np

class CatDetectionMixin:
    def __init__(self, problem_type, **kwargs):
        super().__init__(**kwargs)
        self.ftd = FeatureTypeDetector(
            target_type=problem_type,
            lgb_model_type="unique-based-binned",
            assign_numeric=True,
            detect_numeric_in_string=False
        )

    def fit(self, X, y, X_val, y_val, time_limit, **kwargs):
        self.ftd.fit(pd.concat([X, X_val]), pd.concat([y, y_val]), verbose=True)
        X = self.ftd.transform(X)
        X_val = self.ftd.transform(X_val)
        # TODO: Sometimes the detector can introduce missing values if numeric features extracted from strings are used - need to fix that in the ft detection, for now, leave it here
        self.means = {}
        for col in X.columns:
            if X[col].dtype == float and X[col].isna().any():
                self.means[col] = X[col].mean()
                X[col] = X[col].fillna(self.means[col])
                X_val[col] = X_val[col].fillna(self.means[col])
        return super().fit(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=time_limit, **kwargs)

    def predict_proba(self, X, *, normalize=None, record_time=False, **kwargs):
        X = self.ftd.transform(X)
        for col in X.columns:
            if X[col].dtype == float and X[col].isna().any():
                X[col] = X[col].fillna(self.means[col])
        return super().predict_proba(X=X, normalize=normalize, record_time=record_time, **kwargs)

    def predict(self, X, **kwargs):
        X = self.ftd.transform(X)
        for col in X.columns:
            if X[col].dtype == float and X[col].isna().any():
                X[col] = X[col].fillna(self.means[col])
        return super().predict(X=X, **kwargs)
    

from tabrepo.benchmark.models.ag import RealMLPModel
class RealMLPModelWithCatDetection(CatDetectionMixin, RealMLPModel):
    pass

from tabrepo.benchmark.models.ag import TabMModel
class TabMModelWithCatDetection(CatDetectionMixin, TabMModel):
    pass

from autogluon.tabular.models import CatBoostModel
class CatBoostModelWithCatDetection(CatDetectionMixin, CatBoostModel):
    pass
