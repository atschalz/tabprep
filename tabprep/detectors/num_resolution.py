import numpy as np
import pandas as pd
from regex import D
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from category_encoders import LeaveOneOutEncoder
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from decimal import Decimal
def _precision(x):
    # turn into a Decimal so scientific notation is handled,
    # then look at the negative exponent
    d = Decimal(str(x)).normalize()
    return max(-d.as_tuple().exponent, 0)

from base_preprocessor import BasePreprocessor
class NumResolutionDetector(BasePreprocessor):
    def __init__(self, 
                 target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,                 
                 ):
        super().__init__(target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, 
                         mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)

        if self.target_type=='regression':
            self.metric = lambda y_true, y_pred: -root_mean_squared_error(y_true, y_pred)
            self.lgb_model = lgb.LGBMRegressor
        elif self.target_type=='binary':
            self.metric = lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
            self.lgb_model = lgb.LGBMClassifier

    def fit(self, X_in, y_in):
        # IDEA for numerical resolution detection: Round the values to the value of highest difference between to neighbouring values
        X = X_in.copy()
        y = y_in.copy()

        # Filter candidates
        X_num = X.select_dtypes(include=[float])
        X_num = X_num.loc[:,~(X_num.fillna(-9999999)==X_num.fillna(-9999999).astype(int)).all().values]
        print(f"FP-Numerical features in {data.name}: {X_num.shape[1]}/{X.shape[1]}")
        if X_num.shape[1] == 0:
            print(f"No numerical features in {data.name}. Skipping...")
            self.detection_attempted = False
            return self
        else: 
            self.detection_attempted = True

        self.precisions = {}
        for col in X_num.select_dtypes(include=['float']).columns:
            nonzero = X_num[col].dropna().loc[lambda s: s != 0]
            if nonzero.empty:
                self.precisions[col] = 0
            else:
                self.precisions[col] = nonzero.map(_precision).max()

        self.precisions = pd.Series(self.precisions, dtype=int)

        min_diff = pd.Series({col: pd.Series(X_num[col].dropna().unique()).sort_values().diff().min() for col in X_num.columns if X_num[col].dtype in [float]})
        self.min_diff_precision = min_diff.map(_precision)
        # min_diff_precision = min_diff.apply(lambda x: len(str(x).split('.')[1]))

        # red = pd.concat([self.min_diff_precision, self.precisions], axis=1).diff(axis=1)[1].min()
        # red_col = pd.concat([self.min_diff_precision, self.precisions], axis=1).diff(axis=1)[1].idxmin()
        # print(f'Highest reduction: {red} in {red_col}')
        
        print(f"Found {self.n_X_duplicates:.2%} duplicates in X. {self.n_Xy_duplicates:.2%} of the samples match on the target.")
        # TEST
        X_use = X.copy()
        obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
        X_use[obj_cols] = X_use[obj_cols].astype('category')

        params = {
                    "objective": "binary" if self.target_type=="binary" else "regression",
                    "boosting_type": "gbdt",
                    "n_estimators": 1000,
                    'min_samples_leaf': 1,
                    "max_depth": 5,
                    "verbosity": -1
                }
        
        self.scores['full'] = {}
        self.significances['full'] = {}
        model = self.lgb_model(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        pipe = Pipeline([("model", model)])
        self.scores['full']['lgb'], preds = self.cv_func(clean_feature_names(X_use), y, pipe, return_preds=True)

        # mean adjustment
        method = 'mean'
        for method in ['mean', 'frequent', 'uniform', 'frequent_uniform']:
            self.scores['full'][f'lgb-dupe-{method}'] = self.adjust_predictions(X_use, y, preds, method=method)
            self.significances['full'][f'lgb-dupe-{method}'] = self.significance_test(
                self.scores['full'][f'lgb-dupe-{method}'] - self.scores['full']['lgb']
            )
            if self.significances['full'][f'lgb-dupe-{method}'] < self.alpha:
                print(f"Manually mapping duplicated values significantly improves performance with p={self.significances['full'][f'lgb-dupe-{method}']:.4f}")
        
        ### With duplicateCountAdder
        X_use = X.copy()
        X_use = DuplicateCountAdder().fit_transform(X_use)
        obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
        X_use[obj_cols] = X_use[obj_cols].astype('category')

        model = self.lgb_model(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        pipe = Pipeline([
            # ('duplicate_count_adder', DuplicateCountAdder()),
            ("model", model)
            ])
        self.scores['full']['lgb-dupe-newfeatTTA'], preds = self.cv_func(clean_feature_names(X_use), y, pipe, return_preds=True)


        X_use = X.copy()
        obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
        X_use[obj_cols] = X_use[obj_cols].astype('category')

        pipe = Pipeline([
            # ('duplicate_count_adder', DuplicateCountAdder()),
            ("model", model)
            ])

        ### With weights as a new feature
        self.scores['full']['lgb-dupe-newfeat'], preds = self.cv_func(clean_feature_names(X_use), y, pipe, 
                                                                  return_preds=True, weight_strategy=None, add_cnt=True)
        method = 'mean'
        for method in ['mean', 'frequent', 'uniform', 'frequent_uniform']:
            self.scores['full'][f'lgb-dupe-newfeat-{method}'] = self.adjust_predictions(X_use, y, preds, method=method)
            self.significances['full'][f'lgb-dupe-newfeat-{method}'] = self.significance_test(
                self.scores['full'][f'lgb-dupe-newfeat-{method}'] - self.scores['full']['lgb']
            )

        ### With weights as a new feature
        self.scores['full']['lgb-dupe-loo'], preds = self.cv_func(clean_feature_names(X_use), y, pipe, 
                                                                  return_preds=True, weight_strategy=None, add_loo=True)
        method = 'mean'
        for method in ['mean', 'frequent', 'uniform', 'frequent_uniform']:
            self.scores['full'][f'lgb-dupe-loo-{method}'] = self.adjust_predictions(X_use, y, preds, method=method)
            self.significances['full'][f'lgb-dupe-loo-{method}'] = self.significance_test(
                self.scores['full'][f'lgb-dupe-loo-{method}'] - self.scores['full']['lgb']
            )

        ### With sample weights
        self.scores['full']['lgb-dupe-weighted'], preds = self.cv_func(clean_feature_names(X_use), y, pipe, 
                                                                  return_preds=True, weight_strategy='basic')
        method = 'mean'
        for method in ['mean', 'frequent', 'uniform', 'frequent_uniform']:
            self.scores['full'][f'lgb-dupe-weighted-{method}'] = self.adjust_predictions(X_use, y, preds, method=method)
            self.significances['full'][f'lgb-dupe-weighted-{method}'] = self.significance_test(
                self.scores['full'][f'lgb-dupe-weighted-{method}'] - self.scores['full']['lgb']
            )

        ### With sample weights
        self.scores['full']['lgb-featcnt'], preds = self.cv_func(clean_feature_names(X_use), y, pipe, 
                                                                  return_preds=True, add_feat_cnt=True)
        method = 'mean'
        for method in ['mean', 'frequent', 'uniform', 'frequent_uniform']:
            self.scores['full'][f'lgb-featcnt-{method}'] = self.adjust_predictions(X_use, y, preds, method=method)
            self.significances['full'][f'lgb-featcnt-{method}'] = self.significance_test(
                self.scores['full'][f'lgb-featcnt-{method}'] - self.scores['full']['lgb']
            )


        ### Combine weights, loo, and cnt features
        self.scores['full']['lgb-dupe-alltricks'], preds = self.cv_func(clean_feature_names(X_use), y, pipe, 
                                                                  return_preds=True, weight_strategy='basic', add_cnt=True, add_loo=True)
        method = 'mean'
        for method in ['mean', 'frequent', 'uniform', 'frequent_uniform']:
            self.scores['full'][f'lgb-dupe-alltricks-{method}'] = self.adjust_predictions(X_use, y, preds, method=method)
            self.significances['full'][f'lgb-dupe-alltricks-{method}'] = self.significance_test(
                self.scores['full'][f'lgb-dupe-alltricks-{method}'] - self.scores['full']['lgb']
            )

        # self.scores['full']['lgb-dupe-reverseweighted'], preds = self.cv_func(clean_feature_names(X_use), y, pipe,
        #                                                           return_preds=True, weight_strategy='reverse')

        # TODO: Decide on logic of how to properly handle dupes
        # X_deduplicated = X.drop_duplicates()
        # y_deduplicated = y.loc[X_deduplicated.index]
        # self.dupe_maps = dict(zip(X_deduplicated.astype(str).sum(axis=1).values,y_deduplicated.values))

        ##### Try to find which values can be mapped and which not
        ### FOR DEBUGGING:
        # pd.DataFrame(self.scores).apply(lambda x: np.mean(np.mean(x)),axis=1)
        # comp_df = pd.DataFrame([pred, new_pred, y_test]).transpose()
        # comp_df.columns = ['pred', 'adj', 'true']
        # comp_df.loc[comp_df.pred!=comp_df.adj]

        ##### Try to add sample weights to the model
        # model = self.lgb_model(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        # pipe = Pipeline([("model", model)])
        # self.scores['full']['lgb'], preds = self.cv_func(clean_feature_names(X_use), y, pipe, return_preds=True)

        return self
    
    def transform(self, X_in, y_in=None):
        return X_in

    def _post_fit(self, X_in, y_in=None):
        # TODO: Find out how to include the post_fit in AG
        y = y_in.copy()
        
        y_adj = [float(self.dupe_maps[i]) if i in self.dupe_maps else float(y.iloc[num])  for num, i in enumerate(test_str)]
        y_adj = pd.Series(y_adj, index=pred.index, name=pred.name)

        return y_adj


if __name__ == "__main__":
    from ft_detection import clean_feature_names
    import os
    from utils import *
    import openml
    from ft_detection import clean_feature_names
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'superconduct'
    for benchmark in ['TabArena']: # ["Grinsztajn", "TabArena", "TabZilla"]:
        exp_name = f"EXP_duplicates_{benchmark}"
        if False: #os.path.exists(f"{exp_name}.pkl"):
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
            if dataset_name not in data.name:
                continue
        
            
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
            if target_type=="multiclass":
                # TODO: Fix this hack
                y = (y==y.value_counts().index[0]).astype(int)  # make it binary
                target_type = "binary"
            elif target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
                y = (y==y.value_counts().index[0]).astype(int)  # make it numeric
            else:
                y = y.astype(float)
            
            detector = DuplicateDetector(
                target_type=target_type, 
            )

            detector.fit(X, y)

            results['performance'][data.name] = detector.scores
            results['significances'][data.name] = detector.significances

            print(pd.DataFrame(results['performance'][data.name]).apply(lambda x: np.mean(np.mean(x)),axis=1).sort_values())

        # with open(f"{exp_name}.pkl", "wb") as f:
        #     pickle.dump(results, f)

    