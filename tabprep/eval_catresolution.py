from re import I
from charset_normalizer import detect
import numpy as np
import pandas as pd
from ray import method
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import openml
import pandas as pd
from sympy import im, rem
from tabprep.utils.eval_utils import get_benchmark_dataIDs
from tabprep.utils.tabarena_utils import get_metadata_df
from tabprep.detectors.ft_detection import FeatureTypeDetector

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

from category_encoders import LeaveOneOutEncoder

from tabprep.proxy_models import (
    TargetMeanRegressor,
    TargetMeanClassifier,
    TargetMeanRegressorCut,
    TargetMeanClassifierCut
)
from tabprep.utils.modeling_utils import make_cv_function
from tabprep.utils.eval_utils import p_value_wilcoxon_greater_than_zero
from sklearn.dummy import DummyClassifier, DummyRegressor

import lightgbm as lgb

def make_cv_stratified_on_x(target_type, n_folds=5, verbose=False, early_stopping_rounds=20, vectorized=False):
    """CV creation function for vectorized versions of the TargetEncoderModels."""
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

class CatResolutionDetector(TransformerMixin, BaseEstimator):
    def __init__(self, target_type, 
                 operation_mode='sequential', 
                 max_to_test=100,  # maximum number of unique values to test
                 drop_unique=False, n_folds=5, cv_method='regular'):
        # TODO: Add parameter to decide from which size we do not try to detect anything
        self.target_type = target_type
        self.operation_mode = operation_mode  # 'sequential' or 'full'
        self.max_to_test = max_to_test
        self.drop_unique = drop_unique
        self.n_folds = n_folds
        self.cv_method = cv_method

        if self.cv_method == 'regular':
            self.cv_func = make_cv_function
        elif self.cv_method == 'stratified':
            self.cv_func = make_cv_stratified_on_x

        if self.target_type == 'regression':
            # self.cv_scores_with_early_stopping = make_cv_stratified_on_x("regression", n_folds=n_folds)

            self.target_model = TargetMeanRegressor()
            self.target_cut_model = lambda t: TargetMeanRegressorCut(q_thresh=t)
        elif self.target_type == 'binary':
            # self.cv_scores_with_early_stopping = make_cv_stratified_on_x("binary", n_folds=n_folds, vectorized=True)

            self.target_model = TargetMeanClassifier()
            self.target_cut_model = lambda t: TargetMeanClassifierCut(q_thresh=t)
          
        self.scores = {}
        self.significances = {}
        self.optimal_thresholds = {}
        self.infrequent_values = {}

    def fit(self, X_input, y_input):
        X = X_input.copy()

        for col in X.columns:
            self.scores[col] = {}
            self.infrequent_values[col] = []
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
                cv_func = self.cv_func("regression", n_folds=self.n_folds)
            elif self.target_type == 'binary':
                cv_func = self.cv_func("binary", n_folds=self.n_folds)

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
        
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in X.columns:
            if col not in self.scores:
                continue
            X_new[col] = X_new[col].apply(lambda x: 'infrequent' if x in self.infrequent_values else x)

        return X_new



class IrrelevantCatDetector(TransformerMixin, BaseEstimator):
    def __init__(self, target_type, method='CV', n_folds=5, cv_method='regular'):
        # TODO: Add parameter to decide from which size we do not try to detect anything
        self.target_type = target_type
        self.method = method  # 'CV' or 'LOO'
        self.n_folds = n_folds
        self.cv_method = cv_method

        if self.cv_method == 'regular':
            self.cv_func = make_cv_function
        elif self.cv_method == 'stratified':
            self.cv_func = make_cv_stratified_on_x

        if self.target_type == 'regression':
            self.dummy_model = DummyRegressor(strategy='mean')
            self.target_model = TargetMeanRegressor()
            self.target_cut_model = lambda t: TargetMeanRegressorCut(q_thresh=t)
            self.metric = lambda y_true, y_pred: -root_mean_squared_error(y_true, y_pred)  
        elif self.target_type == 'binary':
            self.dummy_model = DummyClassifier(strategy='prior')
            self.target_model = TargetMeanClassifier()
            self.target_cut_model = lambda t: TargetMeanClassifierCut(q_thresh=t)
            self.metric = lambda y_true, y_pred: -log_loss(y_true, y_pred)  
          
        self.scores = {}
        self.significances = {}
        self.irrelevant_features = []

    def fit(self, X_input, y_input):
        X = X_input.copy()

        for col in X.columns:
            self.significances[col] = {}
            self.scores[col] = {}
            y = y_input.copy()
            x = X[col].copy()

            if self.method == 'CV':
                if self.target_type == 'regression':
                    cv_func = self.cv_func("regression", n_folds=self.n_folds)
                elif self.target_type == 'binary':
                    cv_func = self.cv_func("binary", n_folds=self.n_folds)

                pipe = Pipeline([("model", self.dummy_model)])
                self.scores[col]['dummy'] = cv_func(x.to_frame(), y, pipe)

                pipe = Pipeline([("model", self.target_model)])
                self.scores[col]['mean'] = cv_func(x.astype('category').to_frame(), y, pipe)
                
                self.significances[col][f"mean_beats_dummy"] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col][f'mean'] - self.scores[col]['dummy']
                    )

                if self.significances[col][f"mean_beats_dummy"] > 0.05:
                    self.irrelevant_features.append(col)
            elif self.method == 'LOO':

                dummy_pred = self.dummy_model.fit(x.to_frame(), y).predict(x.to_frame())
                self.scores[col]['dummy'] = self.metric(y, dummy_pred)

                loo_pred = LeaveOneOutEncoder().fit_transform(x.astype('category'), y)[col]
                if self.target_type == 'binary':
                    self.scores[col]['loo'] = np.max([
                        self.metric(y, loo_pred), 
                        self.metric(y, 1-loo_pred)
                        ])
                elif self.target_type == 'regression':
                    self.scores[col]['loo'] = self.metric(y, loo_pred)

            if self.scores[col]['loo'] < self.scores[col]['dummy']:
                self.irrelevant_features.append(col)

        return self

    def transform(self, X):
        if len(self.irrelevant_features)==0:
            return X

        return X.drop(self.irrelevant_features, axis=1)
    


if __name__ == "__main__":


    benchmark = "TabZilla"  # or "TabArena", "TabZilla", "Grinsztajn"
    # dataset_name = 'Amazon_employee_access'  # (['airfoil_self_noise', 'Amazon_employee_access', 'anneal', 'Another-Dataset-on-used-Fiat-500', 'bank-marketing', 'Bank_Customer_Churn', 'blood-transfusion-service-center', 'churn', 'coil2000_insurance_policies', 'concrete_compressive_strength', 'credit-g', 'credit_card_clients_default', 'customer_satisfaction_in_airline', 'diabetes', 'Diabetes130US', 'diamonds', 'E-CommereShippingData', 'Fitness_Club', 'Food_Delivery_Time', 'GiveMeSomeCredit', 'hazelnut-spread-contaminant-detection', 'healthcare_insurance_expenses', 'heloc', 'hiva_agnostic', 'houses', 'HR_Analytics_Job_Change_of_Data_Scientists', 'in_vehicle_coupon_recommendation', 'Is-this-a-good-customer', 'kddcup09_appetency', 'Marketing_Campaign', 'maternal_health_risk', 'miami_housing', 'NATICUSdroid', 'online_shoppers_intention', 'physiochemical_protein', 'polish_companies_bankruptcy', 'APSFailure', 'Bioresponse', 'qsar-biodeg', 'QSAR-TID-11', 'QSAR_fish_toxicity', 'SDSS17', 'seismic-bumps', 'splice', 'students_dropout_and_academic_success', 'taiwanese_bankruptcy_prediction', 'website_phishing', 'wine_quality', 'MIC', 'jm1', 'superconductivity']) 

    tids, dids = get_benchmark_dataIDs(benchmark)  

    remaining_cols = {}

    for tid, did in zip(tids, dids):
        task = openml.tasks.get_task(tid)  # to check if the datasets are available
        data = openml.datasets.get_dataset(did)  # to check if the datasets are available
        # if data.name!=dataset_name:
        #     continue
        # else:
        #     break
        print(data.name)
        X, _, _, _ = data.get_data()
        y = X[data.default_target_attribute]
        X = X.drop(columns=[data.default_target_attribute])
        
        if benchmark == "Grinsztajn" and X.shape[0]>10000:
            X = X.sample(10000, random_state=0)
            y = y.loc[X.index]

        if task.task_type == "Supervised Classification":
            target_type = "binary" if y.nunique() == 2 else "multiclass"
        else:
            target_type = 'regression'
        if target_type=="multiclass":
            # TODO: Fix this hack
            y = (y==y.value_counts().index[0]).astype(int)  # make it binary
            target_type = "binary"
        elif target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
            y = (y==y.value_counts().index[0]).astype(int)  # make it numeric
        else:
            y = y.astype(float)

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        # detector = IrrelevantCatDetector(
        #     target_type=target_type, 
        #     method='LOO',  # 'CV' or 'LOO'
        #     n_folds=5, cv_method='regular'
        #     )
        # detector.fit(X[cat_cols], y)
        # if len(detector.irrelevant_features) > 0:
        #     print(f"Irrelevant columns found ({len(detector.irrelevant_features)}/{len(cat_cols)}): {detector.irrelevant_features}")
        #     print(pd.Series({col: np.mean(detector.scores[col]['loo']) for col in cat_cols}))

        detector = CatResolutionDetector(
            target_type=target_type, 
            operation_mode='sequential',  # 'sequential' or 'full'
            max_to_test=100,  # maximum number of unique values to test
            drop_unique=False, n_folds=5, cv_method='regular'
        )
        detector.fit(X[cat_cols], y)
        thresholds = pd.Series(detector.optimal_thresholds).sort_values(ascending=False)
        if any(thresholds > 0):
            print(thresholds)


        # detector = FeatureTypeDetector(target_type=target_type, 
        #                             interpolation_criterion="match",  # 'win' or 'match'
        #                             lgb_model_type='huge-capacity',
        #                             verbose=False,
        #                             min_q_as_num=3
        #                             )
        
        # # detector.fit(X, y, verbose=False)
        # # print(pd.Series(detector.dtypes).value_counts())
        
        # rem_cols: list = detector.handle_trivial_features(X)
        # print('--'*50)
        # print(f"\rTrivial features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue

        # rem_cols = detector.leave_one_out_test(X, y)
        # print('--'*50)
        # print(f"\rLeave-One-Out features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue


        # rem_cols = detector.get_dummy_mean_scores(X, y)
        # print('--'*50)
        # print(f"\rIrrelevant features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue    
        
        
        # rem_cols = detector.combination_test(X, y, max_binning_configs=3)
        # print('--'*50)
        # print(f"\rCombination features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue
        
        # rem_cols = detector.interpolation_test(X, y, max_degree=3)
        # print('--'*50)
        # print(f"\rInterpolation features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue

        # rem_cols = detector.performance_test(X, y)
        # print('--'*50)
        # print(f"\rLGB-performance features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue

        # remaining_cols[data.name] = rem_cols
        # print(f"{data.name} ({len(rem_cols)}): {rem_cols}")
        # for col in rem_cols:
        #     if col not in detector.dtypes:
        #         detector.dtypes[col] = 'categorical'

        # X = X[rem_cols]
        

        # if len(rem_cols) > 0:
        #     cat_res = CatResolutionDetector(target_type='regression').fit(X, y)
        #     print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))

        # if data.name=='nyc-taxi-green-dec-2016':
        #     break

        # for i, col in enumerate(X.columns):
        #     grouped_interpolation_test(X[col],(y==y[0]).astype(int), 'binary', add_dummy=True if i==0 else False)


        # X, _, _, _ = data.get_data()
        # X = X.drop(columns=[data.default_target_attribute])
        # X = X[[col for col,dtype in detector.dtypes.items() if dtype == 'categorical']]
        # print('Sequential mode:')
        # cat_res = CatResolutionDetector(target_type=target_type, drop_unique=False, 
        #                                 operation_mode='sequential', cv_method='regular'
        #                                 ).fit(X, y)
        # print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))

        # print('Full mode:')
        # cat_res = CatResolutionDetector(target_type=target_type, drop_unique=False, 
        #                                 operation_mode='full', cv_method='regular'
        #                                 ).fit(X, y)
        # print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))

        # print('Sequential mode without uniques:')
        # cat_res = CatResolutionDetector(target_type=target_type, drop_unique=True, 
        #                                 operation_mode='sequential', cv_method='regular'
        #                                 ).fit(X, y)
        # print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))


        