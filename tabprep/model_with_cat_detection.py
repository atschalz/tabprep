from tabrepo.benchmark.models.ag import RealMLPModel
from ft_detection import FeatureTypeDetector
import pandas as pd
import numpy as np

class CatDetectionMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ftd = None

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None, time_limit=None, **kwargs):
        
        self.ftd = FeatureTypeDetector(
            target_type=self.problem_type, # type: ignore
            lgb_model_type="unique-based-binned",
            assign_numeric=True,
            detect_numeric_in_string=False
        )         
        
        self.ftd.fit(X, y, verbose=True)
        # TODO: Implement logic to apply ftd on whole dataset if validation set is provided
        # if X_val is None:
        #     self.ftd.fit(X, y, verbose=True)
        # else:
        #     self.ftd.fit(pd.concat([X, X_val]), pd.concat([y, y_val]), verbose=True)

        self.new_categorical = list(self.ftd.cat_dtype_maps.keys())
        self.new_numeric = [col for col in X.columns if self.ftd.dtypes[col]=="numeric" and self.ftd.orig_dtypes[col]!="numeric"]
        print(f"New categorical: {self.new_categorical}")
        print(f'New numeric: {self.new_numeric}')
        
        X = self.ftd.transform(X)
        
        if X_val is not None:
            X_val = self.ftd.transform(X_val)
        
        return super()._fit(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=time_limit, **kwargs) # type: ignore

    def _preprocess(self, X: pd.DataFrame, **kwargs):
        
        X = self.ftd.transform(X) # type: ignore

        return super()._preprocess(X=X, **kwargs) # type: ignore
    
    def _post_fit(self, **kwargs):
        super()._post_fit(**kwargs) # type: ignore
        self._fit_metadata['new_categorical'] = self.new_categorical # type: ignore
        self._fit_metadata['new_numeric'] = self.new_numeric # type: ignore
        return self

from tabrepo.benchmark.models.ag import RealMLPModel
class RealMLPModelWithCatDetection(CatDetectionMixin, RealMLPModel): # type: ignore
    pass

from tabrepo.benchmark.models.ag import TabMModel
class TabMModelWithCatDetection(CatDetectionMixin, TabMModel): # type: ignore
    pass

from autogluon.tabular.models import CatBoostModel
class CatBoostModelWithCatDetection(CatDetectionMixin, CatBoostModel): # type: ignore
    pass

from autogluon.tabular.models import LGBModel
class LGBModelWithCatDetection(CatDetectionMixin, LGBModel): # type: ignore
    pass

