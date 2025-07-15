import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sympy import rem
from proxy_models import TargetMeanClassifier, TargetMeanRegressor, TargetMeanClassifierCut, TargetMeanRegressorCut
import openml
import pandas as pd
from utils import get_benchmark_dataIDs, get_metadata_df, make_cv_function
from ft_detection import FeatureTypeDetector

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from category_encoders import LeaveOneOutEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from base_preprocessor import BasePreprocessor

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

class IrrelevantCatDetector(BasePreprocessor):
    def __init__(self, target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, random_state=42, verbose=True,
                 method='CV', 
                 cv_method='regular', 
                 **kwargs):
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.method = method  # 'CV' or 'LOO'
        self.cv_method = cv_method

        if self.cv_method == 'regular':
            self.make_cv_function = make_cv_function
        elif self.cv_method == 'stratified':
            self.make_cv_function = make_cv_stratified_on_x

        if self.target_type == 'regression':
            self.dummy_model = DummyRegressor(strategy='mean')
            self.target_model = TargetMeanRegressor()
            self.target_cut_model = lambda t: TargetMeanRegressorCut(q_thresh=t)
            self.metric = lambda y_true, y_pred: -root_mean_squared_error(y_true, y_pred)  
        elif self.target_type == 'binary':
            self.dummy_model = DummyClassifier(strategy='prior')
            self.target_model = TargetMeanClassifier()
            self.target_cut_model = lambda t: TargetMeanClassifierCut(q_thresh=t)
            # TODO: Test metric selection 
            self.metric = lambda y_true, y_pred: -log_loss(y_true, y_pred)  
          
        self.irrelevant_features = []

    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
                # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
            if mode == 'backward': 
                pass
            elif mode == 'forward':
                X_out = X_out.drop(test_cols, axis=1)
        else:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                X_out = X_out.drop(set(test_cols) - {col}, axis=1)
            elif mode == 'forward':
                X_out = X_out.drop(col, axis=1)
        
        return X_out

    def multivariate_performance_test(self, X_cand_in, y_in, 
                                      test_cols, max_cols_use=100):
        suffix = 'DROP'
        drop_cols = super().multivariate_performance_test(X_cand_in, y_in, 
                                      test_cols, suffix=suffix,  max_cols_use=max_cols_use)

        print(f"Drop {len(drop_cols)} columns after multivariate performance test.")
        return drop_cols

        

    def fit(self, X_input, y_input):
        X = X_input.copy()

        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) == 0:
            print("No categorical columns found, skipping detection.")
            return self
        for col in cat_cols:
            if col not in self.scores:
                self.scores[col] = {}
            if col not in self.significances:
                self.significances[col] = {}
            y = y_input.copy()
            x = X[col].copy()

            if self.method == 'CV':
                if self.target_type == 'regression':
                    cv_func = self.make_cv_function("regression", n_folds=self.n_folds)
                elif self.target_type == 'binary':
                    cv_func = self.make_cv_function("binary", n_folds=self.n_folds)

                pipe = Pipeline([("model", self.dummy_model)])
                self.scores[col]['dummy'] = cv_func(x.to_frame(), y, pipe)

                pipe = Pipeline([("model", self.target_model)])
                self.scores[col]['mean'] = cv_func(x.astype('category').to_frame(), y, pipe)
                
                self.significances[col][f"mean_beats_dummy"] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col][f'mean'] - self.scores[col]['dummy']
                    )

                if self.significances[col][f"mean_beats_dummy"] > self.alpha:
                    self.irrelevant_features.append(col)
            elif self.method == 'LOO':

                dummy_pred = self.dummy_model.fit(x.to_frame(), y).predict(x.to_frame())
                self.scores[col]['full_dummy'] = self.metric(y, dummy_pred)

                loo_pred = LeaveOneOutEncoder().fit_transform(x.astype('category'), y)[col]
                if self.target_type == 'binary':
                    self.scores[col]['loo'] = np.max([ # TODO: Verify that using max is correct here
                        self.metric(y, loo_pred), 
                        self.metric(y, 1-loo_pred)
                        ])
                elif self.target_type == 'regression':
                    self.scores[col]['loo'] = self.metric(y, loo_pred)

            if self.scores[col]['loo'] < self.scores[col]['full_dummy']:
                self.irrelevant_features.append(col)
        if len(self.irrelevant_features) == 0:
            print("No irrelevant categorical features found.")
            return self
        self.irrelevant_features = self.multivariate_performance_test(X, y_input, self.irrelevant_features)

        return self

    def transform(self, X):
        if len(self.irrelevant_features)==0:
            return X

        return X.drop(self.irrelevant_features, axis=1)

# if __name__ == "__main__":
#     from ft_detection import clean_feature_names
#     benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
#     for benchmark in ["Grinsztajn", "TabArena", "TabZilla"]:
#         exp_name = f"EXP_dropcat_{benchmark}"
#         if os.path.exists(f"{exp_name}.pkl"):
#             with open(f"{exp_name}.pkl", "rb") as f:
#                 results = pickle.load(f)
#         else:
#             results = {}
#             results['performance'] = {}
#             results['iterations'] = {}
#             results['significances'] = {}

#         tids, dids = get_benchmark_dataIDs(benchmark)  

#         remaining_cols = {}

#         for tid, did in zip(tids, dids):
#             task = openml.tasks.get_task(tid)  # to check if the datasets are available
#             data = openml.datasets.get_dataset(did)  # to check if the datasets are available
#             if data.name in results['performance']:
#                 print(f"Skipping {data.name} as it already exists in results.")
#                 print(pd.Series(results['performance'][data.name]).sort_values(ascending=False))
#                 continue
#             # else:
#             #     break
#             print(data.name)
#             if data.name == 'guillermo':
#                 continue
#             X, _, _, _ = data.get_data()
#             y = X[data.default_target_attribute]
#             X = X.drop(columns=[data.default_target_attribute])
            
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
            
#             cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
#             detector = IrrelevantCatDetector(
#                 target_type=target_type, 
#                 method='LOO',  # 'CV' or 'LOO'
#                 n_folds=5, cv_method='regular'
#                 )
#             detector.fit(X[cat_cols], y)
#             # if len(detector.irrelevant_features) > 0:
#             #     print(f"Irrelevant columns found ({len(detector.irrelevant_features)}/{len(cat_cols)}): {detector.irrelevant_features}")
#             #     print(pd.Series({col: np.mean(detector.scores[col]['loo']) for col in cat_cols}))

#             rem_cols = detector.irrelevant_features                
#             rem_cols = [col for col in rem_cols if X[col].nunique() > 2]  
#             print(f"{data.name} ({len(rem_cols)}): {rem_cols}")

#             if len(rem_cols) == 0:
#                 print(f"Dataset {data.name} has no candidate features, skipping...")
#                 continue
            
#             # if len(set(num_cols)-set(rem_cols))>100:
#             #     print(f"Dataset {data.name} has too many numerical features, downsampling for efficiency.")
#             #     use_cols = np.unique(np.random.choice(num_cols, size=100, replace=False).tolist()+rem_cols).tolist()
#             # else:
#             #     use_cols = np.unique(num_cols+rem_cols).tolist()

#             # X = X[use_cols]
#             # if len(rem_cols) >10:
#             #     print(f"Dataset {data.name} has {len(rem_cols)} features, skipping...")
#             #     continue

#             params = {
#                         "objective": "binary" if target_type=="binary" else "regression",
#                         "boosting_type": "gbdt",
#                         "n_estimators": 1000,
#                         'min_samples_leaf': 2,
#                         "max_depth": 5,
#                         "verbosity": -1
#                     }
#             cv_func = make_cv_scores_with_early_stopping(target_type=target_type,verbose=False, n_folds=10)

#             model = lgb.LGBMClassifier(**params) if target_type=="binary" else lgb.LGBMRegressor(**params)
#             pipe = Pipeline([("model", model)])

#             performances = {}
#             iterations = {}
#             significances = {}
#             X_use = X.copy()
#             all_perf, all_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
#             print(f"ALL: {np.mean(all_perf):.3f}, {np.mean(all_iter):.0f} iterations")

#             X_use = X.copy()
#             X_use = X_use.drop(rem_cols, axis=1)
#             drop_all_perf, drop_all_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
#             print(f"DROP-ALL: {np.mean(drop_all_perf):.3f}, {np.mean(drop_all_iter):.0f} iterations")

#             performances['ALL'] = all_perf
#             performances['DROP-ALL'] = drop_all_perf
#             iterations['ALL'] = all_iter
#             iterations['DROP-ALL'] = drop_all_iter
#             significances['DROP-ALL'] = p_value_wilcoxon_greater_than_zero(drop_all_perf-all_perf)
#             if significances['DROP-ALL'] < 0.05:
#                 print(f"DROP-ALL is significantly better than ALL with p-value {significances['DROP-ALL']:.3f}")


#             for num, col in enumerate(rem_cols):
#                 X_use = X.copy()
#                 X_use = X_use.drop(col,axis=1)
#                 drop_col_perf, drop_col_iter = cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
#                 print(f"{col}-DROP: {np.mean(drop_col_perf):.4f}, {np.mean(drop_col_iter):.0f} iterations")

#                 # performances[f'ALL-NUM'] = str(round(np.mean(num_perf), 3)) + f" ({np.std(num_perf):.3f})"
#                 performances[f'{col}-DROP'] = drop_col_perf
#                 iterations[f'{col}-DROP'] = drop_col_iter
#                 print(f"Column {num+1}/{len(rem_cols)}: {col}", )
#                 significances[f'{col}-DROP'] = p_value_wilcoxon_greater_than_zero(drop_col_perf-all_perf)
#                 if significances[f'{col}-DROP'] < 0.05:
#                     print(f"{col}-DROP is significantly better than ALL-NUM with p-value {significances[f'{col}-DROP']:.3f}")

#             print(pd.DataFrame(performances).mean().sort_values(ascending=False))

#             results['performance'][data.name] = performances
#             results['iterations'][data.name] = iterations
#             results['significances'][data.name] = significances

#             with open(f"{exp_name}.pkl", "wb") as f:
#                 pickle.dump(results, f)


