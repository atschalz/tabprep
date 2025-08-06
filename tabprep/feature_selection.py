import numpy as np
import pandas as pd
from itertools import combinations 
from tabprep.utils import make_cv_function, clean_feature_names
from tabprep.proxy_models import TargetMeanRegressor, TargetMeanClassifier, UnivariateLinearRegressor, UnivariateLogisticClassifier
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.pipeline import Pipeline

from tabprep.base_preprocessor import BasePreprocessor
class FeatureSelector(BasePreprocessor):
    def __init__(self, 
                 target_type, 
                n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                min_drop_scenarios=5,
                min_imp_thresh=0.05, # If the least important feature is used at least this often, we don't attempt feature selection
                 ):
        # TODO: Include possibility to select operators
        # TODO: Might wanna add a hyperparameter to not attempt selection on datasets with too few features
        # TODO: Might wanna add a hyperparameter to limit selection attempts for corner cases
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.min_drop_scenarios = min_drop_scenarios
        self.min_imp_thresh = min_imp_thresh
        self.target_type = target_type
        self.cv_func = make_cv_function(target_type=target_type, early_stopping_rounds=20, vectorized=False, random_state=self.random_state)

        self.cols_to_drop = []

    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
                # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
            if mode == 'backward': 
                pass
            elif mode == 'forward':
                for col in test_cols:
                        if self.target_type=='regression':
                            X_out[col+'_linearized'] = self.linear_features[col].predict(X_out[col].to_frame())
                        else:
                            X_out[col+'_linearized'] = self.linear_features[col].predict_proba(X_out[col].to_frame())[:,1]
        else:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                for use_col in test_cols:
                    if col != use_col:
                        if self.target_type=='regression':
                            X_out[use_col+'_linearized'] = self.linear_features[use_col].predict(X_out[use_col].to_frame())
                        else:
                            X_out[use_col+'_linearized'] = self.linear_features[use_col].predict_proba(X_out[use_col].to_frame())[:,1]
            elif mode == 'forward':
                for use_col in test_cols:
                    if col == use_col:
                        if self.target_type=='regression':
                            X_out[use_col+'_linearized'] = self.linear_features[use_col].predict(X_out[use_col].to_frame())
                        else:
                            X_out[use_col+'_linearized'] = self.linear_features[use_col].predict_proba(X_out[use_col].to_frame())[:,1]

        return X_out

    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        
        if self.target_type != 'regression':
            y = pd.Series(LabelEncoder().fit_transform(y), name=y.name, index = y.index)

        # TODO: Change to get_lgb_params
        params = self.adapt_lgb_params(X[X.nunique().idxmax()])
        params['objective'] = self.target_type
        if self.target_type == 'multiclass':
            params['num_class'] = y.nunique()

        prefix = "LGB-full"
        self.scores[prefix] = {}
        self.significances[prefix] = {}
        X_use = X.copy()
        # TODO: Write a 'preprocess_for_lgb' function using AG
        # Necessary preprocessing
        obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
        X_use[obj_cols] = X_use[obj_cols].astype('category')
        # Adapt data adnd train models
        # TODO: Try a custom LGB model that uses feature bagging heavily!
        model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        pipe = Pipeline([("model", model)])
        self.scores[prefix]['raw'], importances_raw = self.cv_func(clean_feature_names(X_use), y, pipe, return_importances=True)

        df_imp = pd.DataFrame(importances_raw).T
        candidate_cols = df_imp.index[df_imp.sum(axis=1)>0]
        self.cols_to_drop.extend(df_imp.index[df_imp.sum(axis=1)==0].to_list())

        print(f'Found {len(self.cols_to_drop)} cols with 0 importance.')

        X_cand = X_use[candidate_cols].copy()
        df_imp = df_imp.loc[candidate_cols]
        unique_importances = df_imp.mean(axis=1).sort_values(ascending=True).unique()
        # TODO: Test to only attempt feature selection if there are enough features
        # TODO: Add mean percentage at which we don't drop any features

        if (df_imp.mean(axis=1)/df_imp.mean(axis=1).sum()).min() < self.min_imp_thresh:

            for num, thresh in enumerate(unique_importances[:-1]):
                print(f"Trying to drop features with importance below {thresh} ({num+1}/{len(unique_importances)})")
                keep_cols = df_imp.index[df_imp.mean(axis=1).values>thresh]
                self.scores[prefix][f'{len(keep_cols)}-feats'] = self.cv_func(clean_feature_names(X_cand[keep_cols]), y, pipe, return_importances=False)
                self.significances[prefix][f'{len(keep_cols)}-feats_beats_all'] = self.significance_test(
                    self.scores[prefix][f'{len(keep_cols)}-feats']-self.scores[prefix]['raw'],
                )
                self.significances[prefix][f'all-feats_beats_{len(keep_cols)}-feats'] = self.significance_test(
                    self.scores[prefix]['raw']-self.scores[prefix][f'{len(keep_cols)}-feats'],
                )
                # TODO: Adapt significance test to account for cases where the performance is equal
                if np.mean(self.scores[prefix][f'{len(keep_cols)}-feats']) > np.mean(self.scores[prefix]['raw']):
                    continue
                elif self.significances[prefix][f'all-feats_beats_{len(keep_cols)}-feats'] < self.alpha:
                    print(f"Early stopping at {len(keep_cols)} features as the performance is significantly worse than using all features.")
                    break
                elif num+1 < self.min_drop_scenarios:
                    continue
                else:
                    print(f"Early stopping at {len(keep_cols)} features as the performance is not improving over using all features.")
                    break
        else:
            print(f'Least important feature is used at least {self.min_imp_thresh*100}% of the time, not attempting feature selection.')

        score_ser = pd.DataFrame({col: pd.DataFrame(self.scores[col]).mean().sort_values(ascending=False) for col in self.scores})[prefix]
        use_config = score_ser.idxmax()
        if use_config == 'raw':
            pass
        elif self.significances[prefix][f'{use_config}_beats_all'] < self.alpha: 
            curr_n_feats = int(score_ser.index[0].split('-')[0])
            for conf, value in zip(score_ser.index[1:],score_ser.values[1:]):
                if conf == 'raw':
                    break
                elif int(conf.split('-')[0]) < curr_n_feats:
                    continue
                elif self.significances[prefix][f'{conf}_beats_all'] < self.alpha:
                    self.significances[prefix][f"{score_ser.index[0]}_beats_{conf}"] = self.significance_test(
                        self.scores[prefix][score_ser.index[0]] - self.scores[prefix][conf]
                        )
                    if self.significances[prefix][f"{score_ser.index[0]}_beats_{conf}"] > self.alpha:
                        curr_n_feats = int(conf.split('-')[0])

            use_config = f"{curr_n_feats}-feats"
            self.cols_to_drop.extend(df_imp.mean(axis=1).sort_values(ascending=False).index[curr_n_feats:].tolist())
        else:
            pass




        '''
        Algorithm:
        1. Get best mean
        2. Test if its better than the raw data - break if not
        3. If yes, test if one of the less restrictive configs is not significantly worse while still being better than raw
        4. If yes, set it to new current_use and repeat t
        5. If no, set current_use to the best mean
        '''



        # keep_cols = df_imp.index[df_imp.mean(axis=1).values>df_imp.mean(axis=1).min()]        
        # keep_cols = df_imp.index[df_imp.mean(axis=1).values>1]
        # keep_cols = df_imp.index[df_imp.min(axis=1).values>0]

        return self
    
    def transform(self, X):
        X_out = X.copy()
        return X_out.drop(self.cols_to_drop, axis=1, errors='ignore')

if __name__ == "__main__":
    import os
    from tabprep.utils import *
    import openml
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'hiva'
    for benchmark in ['TabArena']: # ["Grinsztajn", "TabArena", "TabZilla"]:
        exp_name = f"EXP_featselect{benchmark}"
        if False: #os.path.exists(f"{exp_name}.pkl"):
            with open(f"{exp_name}.pkl", "rb") as f:
                results = pickle.load(f)
        else:
            results = {}
            results['performance'] = {}
            results['drop_cols'] = {}
            results['significances'] = {}

        tids, dids = get_benchmark_dataIDs(benchmark)  

        remaining_cols = {}

        for tid, did in zip(tids, dids):
            task = openml.tasks.get_task(tid)  # to check if the datasets are available
            data = openml.datasets.get_dataset(did)  # to check if the datasets are available
            # if dataset_name not in data.name:
            #     continue
        
            if data.name in results['performance']:
                print(f"Skipping {data.name} as it already exists in results.")
                print(pd.DataFrame(results['performance'][data.name]).mean().sort_values(ascending=False))
                continue
            # else:
            #     break
            print(data.name)
            if data.name == 'guillermo':
                continue
            X, _, _, _ = data.get_data()
            y = X[data.default_target_attribute]
            X = X.drop(columns=[data.default_target_attribute])
            
            # X = X.sample(n=1000)
            # y = y.loc[X.index]

            if benchmark == "Grinsztajn" and X.shape[0]>10000:
                X = X.sample(10000, random_state=0)
                y = y.loc[X.index]

            if task.task_type == "Supervised Classification":
                target_type = "binary" if y.nunique() == 2 else "multiclass"
            else:
                target_type = 'regression'
            
            detector = FeatureSelector(
                target_type=target_type, 
            )

            detector.fit(X, y)
            print(pd.DataFrame(detector.significances))
            print(pd.DataFrame({col: pd.DataFrame(detector.scores[col]).mean().sort_values() for col in detector.scores}).transpose())
            print(f'Drop {len(detector.cols_to_drop)} cols: {detector.cols_to_drop}')
            results['performance'][data.name] = detector.scores
            results['significances'][data.name] = detector.significances
            results['drop_cols'][data.name] = detector.cols_to_drop

        with open(f"{exp_name}.pkl", "wb") as f:
            pickle.dump(results, f)
        break


