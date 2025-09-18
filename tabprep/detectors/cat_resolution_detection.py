import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sympy import rem
import openml
import pandas as pd

from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor, TargetMeanClassifierCut, TargetMeanRegressorCut
from tabprep.utils.modeling_utils import make_cv_function
from tabprep.utils.eval_utils import get_benchmark_dataIDs, p_value_wilcoxon_greater_than_zero
from tabprep.detectors.ft_detection import FeatureTypeDetector

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

from category_encoders import LeaveOneOutEncoder
import lightgbm as lgb

def make_cv_stratified_on_x(target_type, n_folds=5, verbose=False, early_stopping_rounds=20, vectorized=False):
    """CV creation function for vectorized versions of the TargetEncoderModels."""
    import lightgbm as lgb
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    if target_type=='binary':
        scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
    elif target_type=='regression':
        scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
    else:   
        raise ValueError("target_type must be 'binary' or 'regression'")
    
    def cv_scores_with_early_stopping(X,y, pipeline):
        scores = []
        for train_idx, test_idx in cv.split(X,LabelEncoder().fit_transform(X.iloc[:, 0].astype('category'))):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            final_model = pipeline.named_steps["model"]

            # if it's an LGBM model, pass in eval_set + callbacks
            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                pipeline.fit(
                    X_tr, y_tr,
                    **{
                        "model__eval_set": [(X_te, y_te)],
                        "model__callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                        # "model__verbose": False
                    }
                )
            else:
                # dummy / other estimators: plain fit
                pipeline.fit(X_tr, y_tr)

            # predict
            if target_type == "binary" and hasattr(pipeline, "predict_proba"):
                if vectorized:
                    preds = pipeline.predict_proba(X_te)[:, :, 1]
                else:
                    preds = pipeline.predict_proba(X_te)[:, 1]
            else:
                preds = pipeline.predict(X_te)

            if vectorized:
                scores.append({col: scorer(y_te, preds[:,num]) for num, col in enumerate(X_tr.columns)})
            else:
                scores.append(scorer(y_te, preds))

        return np.array(scores)
    
    return cv_scores_with_early_stopping

from tabprep.detectors.base_preprocessor import BasePreprocessor
class CatResolutionDetector(BasePreprocessor):
    def __init__(self, target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                 operation_mode='sequential', 
                 max_to_test=100,  # maximum number of unique values to test
                 drop_unique=False, cv_method='regular'):
        # TODO: Add parameter to decide from which size we do not try to detect anything
        super().__init__(target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, 
                         mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        
        self.operation_mode = operation_mode  # 'sequential' or 'full'
        self.max_to_test = max_to_test
        self.drop_unique = drop_unique
        self.cv_method = cv_method

        if self.cv_method == 'regular':
            self.make_cv_func = make_cv_function
        elif self.cv_method == 'stratified':
            self.make_cv_func = make_cv_stratified_on_x

        if self.target_type == 'regression':
            # self.cv_scores_with_early_stopping = make_cv_stratified_on_x("regression", n_folds=n_folds)

            self.target_model = TargetMeanRegressor()
            self.target_cut_model = lambda t: TargetMeanRegressorCut(q_thresh=t)
        elif self.target_type == 'binary':
            # self.cv_scores_with_early_stopping = make_cv_stratified_on_x("binary", n_folds=n_folds, vectorized=True)

            self.target_model = TargetMeanClassifier()
            self.target_cut_model = lambda t: TargetMeanClassifierCut(q_thresh=t)
          
        self.optimal_thresholds = {}
        self.infrequent_values = {}

    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
                # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
            if mode == 'backward': 
                pass
            elif mode == 'forward':
                for col in test_cols:
                    X_out[col] = X_out[col].apply(lambda x: 'infrequent' if x in self.infrequent_values[col] else x).astype('category')
        else:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                for col_use in test_cols:
                    if col_use == col:
                        continue
                    X_out[col_use] = X_out[col_use].apply(lambda x: 'infrequent' if x in self.infrequent_values[col_use] else x).astype('category')
            elif mode == 'forward':
                X_out[col] = X_out[col].apply(lambda x: 'infrequent' if x in self.infrequent_values[col] else x).astype('category')
            
        return X_out


    def fit(self, X_input, y_input):
        X = X_input.copy()

        for col in X.columns:
            if col not in self.scores:
                self.scores[col] = {}
            self.infrequent_values[col] = []
            if col not in self.significances:
                self.significances[col] = {}
            self.optimal_thresholds[col] = 0

            y = y_input.copy()
            x = X[col].copy()
            infreq = x.value_counts().sort_values(ascending=True).unique()#[:-1]

            if self.drop_unique:
                # Drop unique values
                x = x[x.isin(x.value_counts()[x.value_counts() > 1].index)]
                print(x.shape)
                y = y.loc[x.index]
                x = x.reset_index(drop=True)
                y = y.reset_index(drop=True)

            if self.target_type == 'regression':
                cv_func = self.make_cv_func("regression", n_folds=self.n_folds)
            elif self.target_type == 'binary':
                cv_func = self.make_cv_func("binary", n_folds=self.n_folds)

            # Target-based stats
            pipe = Pipeline([("model", self.target_model)])
            self.scores[col]['mean'] = cv_func(x.astype('category').to_frame(), y, pipe)
            for t in infreq:
                if t> self.max_to_test:
                    continue
                t = int(t)
                pipe = Pipeline([("model", self.target_cut_model(t))])
                self.scores[col][f'mean-u>{t}'] = cv_func(x.astype('category').to_frame(), y, pipe)

                self.significances[col][f"cut<={t}"] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col][f'mean-u>{t}'] - self.scores[col]['mean']
                    )

                if self.significances[col][f"cut<={t}"] < 0.05:
                    self.optimal_thresholds[col] = t

                    cmap = dict(x.value_counts())
                    self.infrequent_values[col] = [k for k, v in cmap.items() if v <= t]

                elif self.operation_mode == 'sequential':
                    break
        candidate_cols = [key for key,value in self.infrequent_values.items() if len(value)>0]
        if len(candidate_cols) == 0:
            print("No candidate columns found, skipping further processing.")
            return self
        self.multivariate_performance_test(X, y_input, test_cols=candidate_cols, suffix='CUT')
        # X_new[col] = X_new[col].apply(lambda x: 'infrequent' if x in self.infrequent_values else x)

        return self

    def transform(self, X):
        X_new = X.copy()
        for col in X.columns:
            if col not in self.scores:
                continue
            X_new[col] = X_new[col].apply(lambda x: 'infrequent' if x in self.infrequent_values else x)

        return X_new

