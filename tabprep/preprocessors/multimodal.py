import numpy as np
import pandas as pd
from tabprep.detectors.base_preprocessor import BasePreprocessor as old_base
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.preprocessing import TargetEncoder
from tabprep.preprocessors.frequency import FrequencyEncoder
from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator
from tabprep.proxy_models import OOFModePredictor
from tabprep.utils.misc import sample_from_set
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from tabprep.utils.misc import drop_highly_correlated_features
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_categorical_dtype
from tabprep.utils.modeling_utils import clean_feature_names

from tabprep.preprocessors.base import CategoricalBasePreprocessor

from typing import List, Dict, Any, Literal
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.tree import ExtraTreeRegressor, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso

from tabprep.proxy_models import CustomLinearModel, OOFModePredictor

from tabprep.preprocessors.base import BasePreprocessor
class OOFModeEncoder(BasePreprocessor):
    def __init__(self, 
                 target_type:str,
                 n_modes:int=1,
                 n_splits:int=5,
                 random_state:int=42,
                 keep_original: bool = True,
                 ):
        super().__init__(keep_original=keep_original)
        self.target_type = target_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_modes = n_modes
        


    # -----------------------------------------------------------
    def _fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()

        modes =y_in.value_counts().index.tolist()[:self.n_modes]

        self.models = [OOFModePredictor(target_type='binary', 
                                      base_model_cls=CustomLinearModel, 
                                      n_splits=self.n_splits, 
                                      random_state=self.random_state,
                                      mode_value=mode
                                      ) for mode in modes]


        for i in range(self.n_modes):
            self.models[i].fit(X, y)

        return self

    # -----------------------------------------------------------
    def _transform(self, X_in, is_train:bool=False, **kwargs):
        X_out = pd.DataFrame(index=X_in.index)

        for i in range(self.n_modes):
            X_out[f'mode_proximity_{i}'] = self.models[i].predict(X_in, is_train=is_train)

        return X_out