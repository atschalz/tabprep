from statistics import LinearRegression
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sympy import Ordinal, rem
from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor, TargetMeanClassifierCut, TargetMeanRegressorCut, UnivariateLinearRegressor, UnivariateLogisticClassifier
import openml
import pandas as pd
from tabprep.utils.modeling_utils import get_benchmark_dataIDs, get_metadata_df, make_cv_function
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from category_encoders import LeaveOneOutEncoder, OneHotEncoder, TargetEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from tabprep.detectors.base_preprocessor import BasePreprocessor
from sklearn.linear_model import LinearRegression, LogisticRegression

class CatTreatmentDetector(BasePreprocessor):
    '''
    Some thoughts:
    * Target encoding: If there are no interactions, but a strong target effect on its own for the feature
    * OHE: If there are interactions of single values, but not necessarily globally
    * Ordinal: If there is an order in how the samples are presented 
    * float/int/sorted(str): If the given encoding has a meaning for itself

    Process could be: 
        1. determine the strength of the target relationship using the TargetMeanPredictor
        2. If possible, transform the feature to a float and determine the strength of the target relationship using linear & combination tests
        3. Ordinally encode the feature and determine the strength of the target relationship using linear & combination tests (TODO: Sort the feature by frequency prior to ordinal encoding to verify stuff)
        4. Sort the feature and determine the strength of the target relationship using linear & combination tests
        5. For OHE, we can check the performance of a linear model after OHE?
    '''
    def __init__(self, target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                 min_cardinality=6,
                 ohe_method='ohe', # ['ohe', 'te', 'loo']
                 lgb_model_type='default',
                 **kwargs):
        super().__init__(target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, 
                         mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.min_cardinality = min_cardinality # TODO: Think about what to do with low-cardinality features
        self.ohe_method = ohe_method
        self.lgb_model_type = lgb_model_type  # default

        self.reassign_cols = []

        if self.target_type == 'regression':
            self.dummy_model = DummyRegressor(strategy='mean')
            self.target_model = TargetMeanRegressor()
            self.univ_linear_model = UnivariateLinearRegressor()
            self.linear_model = LinearRegression()
            self.target_cut_model = lambda t: TargetMeanRegressorCut(q_thresh=t)
            self.metric = lambda y_true, y_pred: -root_mean_squared_error(y_true, y_pred)  
        elif self.target_type == 'binary':
            self.dummy_model = DummyClassifier(strategy='prior')
            self.target_model = TargetMeanClassifier()
            self.univ_linear_model = UnivariateLogisticClassifier()
            self.linear_model = LogisticRegression()
            self.target_cut_model = lambda t: TargetMeanClassifierCut(q_thresh=t)
            # TODO: Test metric selection 
            self.metric = lambda y_true, y_pred: -log_loss(y_true, y_pred)  

        self.iterations = {}

    def filter_candidates_by_distinctiveness(self, X):
            candidate_cols = []
            for col in X.columns:
                x_new = X[col].map(X[col].value_counts().to_dict())
                if all((pd.crosstab(X[col],x_new)>0).sum()==1):
                    continue
                else:
                    candidate_cols.append(col)

            return candidate_cols

    def transform_cat_col(self, X, col):
        X_out = X.copy()
        if self.use_cat_col_as[col] == 'convert':
            X_out[col] = pd.to_numeric(X_out[col], errors='coerce')
        elif self.use_cat_col_as[col] == 'ordinal':
            X_out[col] = OrdinalEncoder().fit_transform(X_out[[col]]).flatten()
        elif self.use_cat_col_as[col] == 'sorted':
            u_sorted = X_out[col].sort_values().unique()
            sorted_map = dict(zip(u_sorted, range(len(u_sorted))))
            X_out[col] = X_out[col].map(sorted_map)
        elif self.use_cat_col_as[col] == 'mean':
            X_out[col] = OrdinalEncoder().fit_transform(X_out[[col]]).flatten()
            # pass # Use the model default method
        elif self.use_cat_col_as[col] == 'ohe':
            # X_ohe = OneHotEncoder(handle_unknown='value', handle_missing='indicator').fit_transform(X_out[[col]])
            # X_ohe = pd.DataFrame(X_ohe, columns=[f"{col}_{c}" for c in X_ohe.columns])
            # X_out = pd.concat([X_out, X_ohe], axis=1)
            # X_out = X_out.drop(col, axis=1)
            pass
        
        return X_out


    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
                # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
            if mode == 'backward': 
                pass
            elif mode == 'forward':
                for col in test_cols:
                    X_out = self.transform_cat_col(X_out, col)                    
        else:
            X_out = X_cand_in.copy()
            for col_use in test_cols:
                if mode == 'backward' and col==col_use:
                    continue  # Convert back to category
                elif mode == 'forward' and col!=col_use:
                    continue
                
                X_out = self.transform_cat_col(X_out, col_use)                    
        
        return X_out

    # TODO: Consider adding an interaction test
    def fit(self, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()

        y = self.adjust_target_format(y)

        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        
        cat_cols = [col for col in cat_cols if X[col].nunique() >= self.min_cardinality]
        if len(cat_cols) == 0:
            self.detection_attempted = False
            return self
        self.scores = {col: {} for col in cat_cols}
        self.significances = {col: {} for col in cat_cols}
        # 1. Get the dummy and target mean scores for the categorical features
        self.get_dummy_mean_scores(X[cat_cols], y)
        
        self.use_cat_col_as = {} # numeric, ordinal, sorted, ohe, linear-TE, linear-LOO
        candidate_cols = []
        for num, col in enumerate(cat_cols):       
            print(f"Processing {num}/{len(cat_cols)}: {col}")
            # IDEAS:
            # - Drop a feature if dummy>all conditions
            # - Use Ordinal encoding (or original values or sorted) if 'mean' is not significantly better     
            # - Need to think deeply about when and why OHE is better or worse than mean
                # Likely, if it is better, this might mean that it effectively avoids making overly confident predictions on unseen values by adding the constant
                # If it is worse, this might mean that we have a very predictive mean for the feature and we should use target encoding over OHE
            # - We could also try to define a percentage starting from which we use ordinal encoding, i.e. if the linear model is mot worse than x%
                # Another approach would be to use an absolute value, depending on how high the highest performance is
            # - BUT: It might as well make sense to also include the other interpolation methods as well as a combination test before deciding.
            

            self.linear_to_use = None
            # 2. Get the linear-float model
            num_convertible = pd.to_numeric(X[col].dropna(), errors='coerce').notna().all()
            if num_convertible:
                X_float = pd.to_numeric(X[col], errors='coerce')
                self.scores[col]['linear-convert'] = self.single_interpolation_test(X_float, y, interpolation_method='linear')

                self.significances[col]['mean_beats_linear-convert'] = self.significance_test(
                    self.scores[col]['mean'] - self.scores[col]['linear-convert'], 
                )
                self.linear_to_use = 'linear-convert'

            # 3. Get the ordinal model
            # TODO: Remove leak
            X_ordinal = pd.Series(OrdinalEncoder().fit_transform(X[[col]]).flatten(), name=col)
            if num_convertible:
                if any(X_ordinal.values != X_float.values):
                    self.scores[col]['linear-ordinal'] = self.single_interpolation_test(X_ordinal, y, interpolation_method='linear')
                    if np.mean(self.scores[col]['linear-ordinal']) > np.mean(self.scores[col]['linear-convert']):
                        self.linear_to_use = 'linear-ordinal'
            else:
                self.scores[col]['linear-ordinal'] = self.single_interpolation_test(X_ordinal, y, interpolation_method='linear')
                self.linear_to_use = 'linear-ordinal'                 
            if 'linear-ordinal' in self.scores[col]:
                self.significances[col]['mean_beats_linear-ordinal'] = self.significance_test(
                    self.scores[col]['mean'] - self.scores[col]['linear-ordinal'], 
                )

            # 4. Get the sorted model
            u_sorted = X[col].sort_values().unique()
            sorted_map = dict(zip(u_sorted, range(len(u_sorted))))
            X_sorted = X[col].map(sorted_map)
            if any(X_sorted.values != X_ordinal.values):
                if not num_convertible or (num_convertible and any(X_sorted != X_float)):
                    self.scores[col]['linear-sorted'] = self.single_interpolation_test(X_sorted, y, interpolation_method='linear')
                if np.mean(self.scores[col]['linear-sorted']) > np.mean(self.scores[col][self.linear_to_use]):
                    self.linear_to_use = 'linear-sorted'
            if 'linear-sorted' in self.scores[col]:
                self.significances[col]['mean_beats_linear-sorted'] = self.significance_test(
                    self.scores[col]['mean'] - self.scores[col]['linear-sorted'], 
                )

            self.use_cat_col_as[col] = self.linear_to_use.split('-')[1]
            if self.significances[col][f'mean_beats_{self.linear_to_use}'] > self.alpha:
                # self.use_cat_col_as[col] = self.linear_to_use.split('-')[1]
                # continue
                candidate_cols.append(col)
            # else:
            #     self.use_cat_col_as[col] = 'mean'

            # 5. Get the OHE model
            # TODO: Use a linear model version that is fast on sparse data
            # self.scores[col]['ohe'] = self.cv_func(X[[col]].copy(), y,
            #                                        Pipeline([
            #                                            ("ohe",  OneHotEncoder(handle_unknown='value', handle_missing='indicator')),
            #                                            ("model", self.linear_model)
            #                                        ]))
            # if np.mean(self.scores[col]['mean']).round(8) < np.mean(self.scores[col]['ohe']).round(8):
            #     self.use_cat_col_as[col] = 'ohe'
            # else:
            #     self.use_cat_col_as[col] = 'mean'

            # TODO: Check OHE>Mean as a method that indicates to add a LOO encoding feature    
            # TODO: But actually it indicatest that we shouldn't be using low-frequency categories

            # 5. Get the TE-linear models
            # TODO: Check whether TE/LOO can replace OHE
            # self.scores[col]['linear-TE'] = self.cv_func(X[[col]].copy(), y,
            #                                        Pipeline([
            #                                            ("te",  TargetEncoder(handle_unknown='value', handle_missing='indicator')),
            #                                            ("model", self.linear_model)
            #                                        ]))
            
            # self.scores[col]['linear-loo'] = self.cv_func(X[[col]].copy(), y,
            #                                        Pipeline([
            #                                            ("loo",  LeaveOneOutEncoder(handle_unknown='value', handle_missing='indicator')),
            #                                            ("model", self.linear_model)
            #                                        ]))


            # TODO: Investigate whether both, adding as num and cat can make sense
            # TODO: Investigate whether always treating columns not significantly better than mean as ordinal is a good idea 

        # self.significances['linear_beats_mean'] = self.scores[col]['linear'].mean() > self.scores[col]['mean'].mean()

            print(col, pd.DataFrame(self.scores[col]).mean().sort_values())
            print()
        self.detection_attempted = True

        # if len(candidate_cols) > 0:
        #     self.multivariate_performance_test(X, y_input, candidate_cols, suffix="ADDFREQ")
        #     self.detection_attempted = True
        # else:
        #     self.detection_attempted = False
        print(pd.DataFrame({col: pd.DataFrame(self.scores[col]).mean().sort_values() for col in self.scores}).transpose())
        print(pd.Series(self.use_cat_col_as))
        # TODO: Add performance test iterating over all possible choices that are suggested for all of the features, even if its only suggested for one feature.
        # TODO: Consider dropping features that are not better than dummy no matter how we treat the feature
        
        # candidate_cols = [col for col in self.use_cat_col_as if self.use_cat_col_as[col] != 'mean']

        if len(candidate_cols) > 0:
            
            self.reassign_cols = self.multivariate_performance_test(X, y, cat_cols, suffix="transform", individual_test_condition = 'significant')
        else:
            self.reassign_cols = []
        
        
        print(pd.DataFrame({col: pd.DataFrame(self.scores[col]).mean().sort_values() for col in self.scores}).transpose())
        print(pd.Series(self.use_cat_col_as))

        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.reassign_cols:
            X_out = self.transform_cat_col(X_out, col)

        return X_out

# if __name__ == "__main__":
#     import os
#     from utils import *
#     import openml
#     import sys
#     import pickle
#     print(os.getcwd())
#     sys.path.append('/home/ubuntu/cat_detection/tabprep')

#     benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
#     dataset_name = 'Diabe'
#     for benchmark in ['TabArena', "Grinsztajn", "TabZilla"]:
#         exp_name = f"EXP_cat_treat_{benchmark}"
#         if False: #os.path.exists(f"{exp_name}.pkl"):
#             with open(f"{exp_name}.pkl", "rb") as f:
#                 results = pickle.load(f)
#         else:
#             results = {} 
#             results['performance'] = {}
#             results['reassign'] = {}
#             results['significances'] = {}
#             results['reassign_to'] = {}

#         tids, dids = get_benchmark_dataIDs(benchmark)  

#         remaining_cols = {}

#         for tid, did in zip(tids, dids):
#             task = openml.tasks.get_task(tid)  # to check if the datasets are available
#             data = openml.datasets.get_dataset(did)  # to check if the datasets are available
#             # if dataset_name not in data.name:
#             #     continue
        
            
#             if data.name in results['performance']:
#                 print(f"Skipping {data.name} as it already exists in results.")
#                 print(pd.DataFrame(results['performance'][data.name]).apply(lambda x: np.mean(np.mean(x)),axis=1))
#                 continue
#             # else:
#             #     break
#             print(data.name)
#             if data.name == 'guillermo':
#                 continue
#             X, _, _, _ = data.get_data()
#             y = X[data.default_target_attribute]
#             X = X.drop(columns=[data.default_target_attribute])
            
#             # X = X.sample(n=1000)
#             # y = y.loc[X.index]

#             if benchmark == "Grinsztajn" and X.shape[0]>10000:
#                 X = X.sample(10000, random_state=0)
#                 y = y.loc[X.index]

#             if task.task_type == "Supervised Classification":
#                 target_type = "binary" if y.nunique() == 2 else "multiclass"
#             else:
#                 target_type = 'regression'
#             if target_type=="multiclass":
#                 # TODO: Fix this hack
#                 y = (y==y.value_counts().index[0]).astype(int)  # make it binary
#                 target_type = "binary"
#             elif target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
#                 y = (y==y.value_counts().index[0]).astype(int)  # make it numeric
#             else:
#                 y = y.astype(float)
            
#             detector = CatTreatmentDetector(
#                 target_type=target_type,
#                 lgb_model_type='default',
#                 min_cardinality=3,
#             )

#             detector.fit(X, y)
#             if detector.detection_attempted:
#                 results['performance'][data.name] = detector.scores
#                 results['significances'][data.name] = detector.significances
#                 results['reassign'][data.name] = detector.reassign_cols
#                 results['reassign_to'][data.name] = detector.use_cat_col_as
#                 print(pd.DataFrame(results['performance'][data.name]).apply(lambda x: np.mean(np.mean(x)),axis=1).sort_values())
#                 print()
#         with open(f"{exp_name}.pkl", "wb") as f:
#             pickle.dump(results, f)

    