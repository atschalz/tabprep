import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sympy import rem
from tabprep.utils import *
from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor, TargetMeanClassifierCut, TargetMeanRegressorCut
import openml
import pandas as pd
from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
from ft_detection import FeatureTypeDetector

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

from category_encoders import LeaveOneOutEncoder

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
                scores.append({col: scorer(y_te, preds[:,num]) for num, col in enumerate(X_df.columns)})
            else:
                scores.append(scorer(y_te, preds))

        return np.array(scores)
    
    return cv_scores_with_early_stopping

from base_preprocessor import BasePreprocessor
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



# if __name__ == "__main__":


#     benchmark = "TabZilla"  # or "TabArena", "TabZilla", "Grinsztajn"
#     # dataset_name = 'Amazon_employee_access'  # (['airfoil_self_noise', 'Amazon_employee_access', 'anneal', 'Another-Dataset-on-used-Fiat-500', 'bank-marketing', 'Bank_Customer_Churn', 'blood-transfusion-service-center', 'churn', 'coil2000_insurance_policies', 'concrete_compressive_strength', 'credit-g', 'credit_card_clients_default', 'customer_satisfaction_in_airline', 'diabetes', 'Diabetes130US', 'diamonds', 'E-CommereShippingData', 'Fitness_Club', 'Food_Delivery_Time', 'GiveMeSomeCredit', 'hazelnut-spread-contaminant-detection', 'healthcare_insurance_expenses', 'heloc', 'hiva_agnostic', 'houses', 'HR_Analytics_Job_Change_of_Data_Scientists', 'in_vehicle_coupon_recommendation', 'Is-this-a-good-customer', 'kddcup09_appetency', 'Marketing_Campaign', 'maternal_health_risk', 'miami_housing', 'NATICUSdroid', 'online_shoppers_intention', 'physiochemical_protein', 'polish_companies_bankruptcy', 'APSFailure', 'Bioresponse', 'qsar-biodeg', 'QSAR-TID-11', 'QSAR_fish_toxicity', 'SDSS17', 'seismic-bumps', 'splice', 'students_dropout_and_academic_success', 'taiwanese_bankruptcy_prediction', 'website_phishing', 'wine_quality', 'MIC', 'jm1', 'superconductivity']) 

#     tids, dids = get_benchmark_dataIDs(benchmark)  

#     remaining_cols = {}

#     for tid, did in zip(tids, dids):
#         task = openml.tasks.get_task(tid)  # to check if the datasets are available
#         data = openml.datasets.get_dataset(did)  # to check if the datasets are available
#         # if data.name!=dataset_name:
#         #     continue
#         # else:
#         #     break
#         print(data.name)
#         X, _, _, _ = data.get_data()
#         y = X[data.default_target_attribute]
#         X = X.drop(columns=[data.default_target_attribute])
        
#         if benchmark == "Grinsztajn" and X.shape[0]>10000:
#             X = X.sample(10000, random_state=0)
#             y = y.loc[X.index]

#         if task.task_type == "Supervised Classification":
#             target_type = "binary" if y.nunique() == 2 else "multiclass"
#         else:
#             target_type = 'regression'
#         if target_type=="multiclass":
#             # TODO: Fix this hack
#             y = (y==y.value_counts().index[0]).astype(int)  # make it binary
#             target_type = "binary"
#         elif target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
#             y = (y==y.value_counts().index[0]).astype(int)  # make it numeric
#         else:
#             y = y.astype(float)

#         cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
#         # detector = IrrelevantCatDetector(
#         #     target_type=target_type, 
#         #     method='LOO',  # 'CV' or 'LOO'
#         #     n_folds=5, cv_method='regular'
#         #     )
#         # detector.fit(X[cat_cols], y)
#         # if len(detector.irrelevant_features) > 0:
#         #     print(f"Irrelevant columns found ({len(detector.irrelevant_features)}/{len(cat_cols)}): {detector.irrelevant_features}")
#         #     print(pd.Series({col: np.mean(detector.scores[col]['loo']) for col in cat_cols}))

#         detector = CatResolutionDetector(
#             target_type=target_type, 
#             operation_mode='sequential',  # 'sequential' or 'full'
#             max_to_test=100,  # maximum number of unique values to test
#             drop_unique=False, n_folds=5, cv_method='regular'
#         )
#         detector.fit(X[cat_cols], y)
#         thresholds = pd.Series(detector.optimal_thresholds).sort_values(ascending=False)
#         if any(thresholds > 0):
#             print(thresholds)


#         # detector = FeatureTypeDetector(target_type=target_type, 
#         #                             interpolation_criterion="match",  # 'win' or 'match'
#         #                             lgb_model_type='huge-capacity',
#         #                             verbose=False,
#         #                             min_q_as_num=3
#         #                             )
        
#         # # detector.fit(X, y, verbose=False)
#         # # print(pd.Series(detector.dtypes).value_counts())
        
#         # rem_cols: list = detector.handle_trivial_features(X)
#         # print('--'*50)
#         # print(f"\rTrivial features removed: {X.shape[1]-len(rem_cols)}\r")
#         # X = X[rem_cols]
#         # if len(rem_cols) == 0:
#         #     remaining_cols[data.name] = []
#         #     continue

#         # rem_cols = detector.leave_one_out_test(X, y)
#         # print('--'*50)
#         # print(f"\rLeave-One-Out features removed: {X.shape[1]-len(rem_cols)}\r")
#         # X = X[rem_cols]
#         # if len(rem_cols) == 0:
#         #     remaining_cols[data.name] = []
#         #     continue


#         # rem_cols = detector.get_dummy_mean_scores(X, y)
#         # print('--'*50)
#         # print(f"\rIrrelevant features removed: {X.shape[1]-len(rem_cols)}\r")
#         # X = X[rem_cols]
#         # if len(rem_cols) == 0:
#         #     remaining_cols[data.name] = []
#         #     continue    
        
        
#         # rem_cols = detector.combination_test(X, y, max_binning_configs=3)
#         # print('--'*50)
#         # print(f"\rCombination features removed: {X.shape[1]-len(rem_cols)}\r")
#         # X = X[rem_cols]
#         # if len(rem_cols) == 0:
#         #     remaining_cols[data.name] = []
#         #     continue
        
#         # rem_cols = detector.interpolation_test(X, y, max_degree=3)
#         # print('--'*50)
#         # print(f"\rInterpolation features removed: {X.shape[1]-len(rem_cols)}\r")
#         # X = X[rem_cols]
#         # if len(rem_cols) == 0:
#         #     remaining_cols[data.name] = []
#         #     continue

#         # rem_cols = detector.performance_test(X, y)
#         # print('--'*50)
#         # print(f"\rLGB-performance features removed: {X.shape[1]-len(rem_cols)}\r")
#         # X = X[rem_cols]
#         # if len(rem_cols) == 0:
#         #     remaining_cols[data.name] = []
#         #     continue

#         # remaining_cols[data.name] = rem_cols
#         # print(f"{data.name} ({len(rem_cols)}): {rem_cols}")
#         # for col in rem_cols:
#         #     if col not in detector.dtypes:
#         #         detector.dtypes[col] = 'categorical'

#         # X = X[rem_cols]
        

#         # if len(rem_cols) > 0:
#         #     cat_res = CatResolutionDetector(target_type='regression').fit(X, y)
#         #     print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))

#         # if data.name=='nyc-taxi-green-dec-2016':
#         #     break

#         # for i, col in enumerate(X.columns):
#         #     grouped_interpolation_test(X[col],(y==y[0]).astype(int), 'binary', add_dummy=True if i==0 else False)


#         # X, _, _, _ = data.get_data()
#         # X = X.drop(columns=[data.default_target_attribute])
#         # X = X[[col for col,dtype in detector.dtypes.items() if dtype == 'categorical']]
#         # print('Sequential mode:')
#         # cat_res = CatResolutionDetector(target_type=target_type, drop_unique=False, 
#         #                                 operation_mode='sequential', cv_method='regular'
#         #                                 ).fit(X, y)
#         # print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))

#         # print('Full mode:')
#         # cat_res = CatResolutionDetector(target_type=target_type, drop_unique=False, 
#         #                                 operation_mode='full', cv_method='regular'
#         #                                 ).fit(X, y)
#         # print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))

#         # print('Sequential mode without uniques:')
#         # cat_res = CatResolutionDetector(target_type=target_type, drop_unique=True, 
#         #                                 operation_mode='sequential', cv_method='regular'
#         #                                 ).fit(X, y)
#         # print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))


        

if __name__ == "__main__":
    from ft_detection import clean_feature_names
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    for benchmark in ["Grinsztajn", "TabArena", "TabZilla"]:
        exp_name = f"EXP_catresolution_perf_{benchmark}"
        if os.path.exists(f"{exp_name}.pkl"):
            with open(f"{exp_name}.pkl", "rb") as f:
                results = pickle.load(f)
        else:
            results = {}
            results['performance'] = {}
            results['iterations'] = {}
            results['significances'] = {}

        tids, dids = get_benchmark_dataIDs(benchmark)  

        remaining_cols = {}

        for tid, did in zip(tids, dids):
            task = openml.tasks.get_task(tid)  # to check if the datasets are available
            data = openml.datasets.get_dataset(did)  # to check if the datasets are available
            if data.name in results['performance']:
                print(f"Skipping {data.name} as it already exists in results.")
                print(pd.Series(results['performance'][data.name]).sort_values(ascending=False))
                continue
            # else:
            #     break
            print(data.name)
            if data.name == 'guillermo':
                continue
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
            cat_cols = [col for col in cat_cols if X[col].nunique() > 2]
            if len(cat_cols) == 0:
                print(f"Dataset {data.name} has no candidate features, skipping...")
                continue
            
            detector = CatResolutionDetector(
                target_type=target_type, 
                operation_mode='sequential',  # 'sequential' or 'full'
                max_to_test=100,  # maximum number of unique values to test
                drop_unique=False, n_folds=5, cv_method='regular'
            )
            detector.fit(X[cat_cols], y)
            thresholds = pd.Series(detector.optimal_thresholds).sort_values(ascending=False)
            
            # print(pd.DataFrame(detector.scores).mean().sort_values())
            
            
            # rem_cols = thresholds[thresholds > 0].index.tolist()                
            # # rem_cols = [col for col in rem_cols if X[col].nunique() > 2]  
            # print(f"{data.name} ({len(rem_cols)}): {rem_cols}")

            # if len(rem_cols) == 0:
            #     print(f"Dataset {data.name} has no candidate features, skipping...")
            #     continue
            
            # # if len(set(num_cols)-set(rem_cols))>100:
            # #     print(f"Dataset {data.name} has too many numerical features, downsampling for efficiency.")
            # #     use_cols = np.unique(np.random.choice(num_cols, size=100, replace=False).tolist()+rem_cols).tolist()
            # # else:
            # #     use_cols = np.unique(num_cols+rem_cols).tolist()

            # # X = X[use_cols]
            # # if len(rem_cols) >10:
            # #     print(f"Dataset {data.name} has {len(rem_cols)} features, skipping...")
            # #     continue

            # params = {
            #             "objective": "binary" if target_type=="binary" else "regression",
            #             "boosting_type": "gbdt",
            #             "n_estimators": 1000,
            #             'min_samples_leaf': 2,
            #             "max_depth": 5,
            #             "verbosity": -1
            #         }
            # cv_func = make_cv_scores_with_early_stopping(target_type=target_type,verbose=False, n_folds=10)

            # model = lgb.LGBMClassifier(**params) if target_type=="binary" else lgb.LGBMRegressor(**params)
            # pipe = Pipeline([("model", model)])

            # performances = {}
            # iterations = {}
            # significances = {}
            # X_use = X.copy()
            # all_perf, all_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
            # print(f"ALL: {np.mean(all_perf):.3f}, {np.mean(all_iter):.0f} iterations")

            # X_use = X.copy()
            # for col in rem_cols:
            #     X_use[col] = X_use[col].apply(lambda x: 'infrequent' if x in detector.infrequent_values[col] else x)
            #     X_use = X_use.astype('category')
            # thresh_all_perf, thresh_all_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
            # print(f"THRESHOLD-ALL: {np.mean(thresh_all_perf):.3f}, {np.mean(thresh_all_iter):.0f} iterations")

            # performances['ALL'] = all_perf
            # performances['THRESHOLD-ALL'] = thresh_all_perf
            # iterations['ALL'] = all_iter
            # iterations['THRESHOLD-ALL'] = thresh_all_iter
            # significances['THRESHOLD-ALL'] = p_value_wilcoxon_greater_than_zero(thresh_all_perf-all_perf)
            # if significances['THRESHOLD-ALL'] < 0.05:
            #     print(f"THRESHOLD-ALL is significantly better than ALL with p-value {significances['THRESHOLD-ALL']:.3f}")


            # for num, col in enumerate(rem_cols):
            #     X_use = X.copy()
            #     # TODO: Make sure to prevent leaks
            #     X_use[col] = X_use[col].apply(lambda x: 'infrequent' if x in detector.infrequent_values[col] else x)
            #     X_use = X_use.astype('category')
            #     drop_col_perf, drop_col_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
            #     print(f"{col}-thresh{thresholds[col]}: {np.mean(drop_col_perf):.4f}, {np.mean(drop_col_iter):.0f} iterations")

            #     # performances[f'ALL-NUM'] = str(round(np.mean(num_perf), 3)) + f" ({np.std(num_perf):.3f})"
            #     performances[f'{col}-thresh{thresholds[col]}'] = drop_col_perf
            #     iterations[f'{col}-thresh{thresholds[col]}'] = drop_col_iter
            #     print(f"Column {num+1}/{len(rem_cols)}: {col}", )
            #     significances[col] = p_value_wilcoxon_greater_than_zero(drop_col_perf-all_perf)
            #     if significances[col] < 0.05:
            #         print(f"{col}-thresh{thresholds[col]} is significantly better than ALL-NUM with p-value {significances[col]:.3f}")

            # print(pd.DataFrame(performances).mean().sort_values(ascending=False))

            # results['performance'][data.name] = performances
            # results['iterations'][data.name] = iterations
            # results['significances'][data.name] = significances

            # with open(f"{exp_name}.pkl", "wb") as f:
            #     pickle.dump(results, f)


