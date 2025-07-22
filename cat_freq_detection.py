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


class CatFreqDetector(BasePreprocessor):
    def __init__(self, target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                 min_cardinality=6,
                 **kwargs):
        super().__init__(target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, 
                         mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.min_cardinality = min_cardinality

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

        self.freq_maps = {}
          
    def get_freq_feature(self, x):
        if x.name+"_freq" in self.freq_maps:
            x_new = x.map(self.freq_maps[x.name])
        else:
            self.freq_maps[x.name] = x.value_counts().to_dict()
            x_new = x.map(self.freq_maps[x.name])
        return x_new

    def filter_candidates_by_distinctiveness(self, X):
            candidate_cols = []
            for col in X.columns:
                x_new = X[col].map(X[col].value_counts().to_dict())
                if all((pd.crosstab(X[col],x_new)>0).sum()==1):
                    continue
                else:
                    candidate_cols.append(col)

            return candidate_cols

    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
                # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
            if mode == 'backward': 
                pass
            elif mode == 'forward':
                for col in test_cols:
                    new_x = self.get_freq_feature(X_out[col])
                    X_out[col+"_freq"] = new_x
                
        else:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                for col_use in test_cols:
                    if col_use==col:
                        continue
                    new_x = self.get_freq_feature(X_out[col_use])
                    X_out[col_use+"_freq"] = new_x
            elif mode == 'forward':
                new_x = self.get_freq_feature(X_out[col])
                X_out[col+"_freq"] = new_x

        return X_out


    def fit(self, X_input, y_input):
        X = X_input.copy()

        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        
        cat_cols = [col for col in cat_cols if X[col].nunique() >= self.min_cardinality]
        if len(cat_cols) == 0:
            self.detection_attempted = False
            return self

        # Filter candidates by distinctiveness
        candidate_cols = self.filter_candidates_by_distinctiveness(X[cat_cols])

        if len(candidate_cols) > 0:
            self.new_cols = self.multivariate_performance_test(X, y_input, candidate_cols, suffix="ADDFREQ")
            self.detection_attempted = True
        else:
            self.detection_attempted = False

        return self

    def transform(self, X):
        X_out = X.copy()
        if self.detection_attempted:
            for col in self.new_cols:
                X_out[col+"_freq"] = self.get_freq_feature(X_out[col])

        return X_out

if __name__ == "__main__":
    from ft_detection import clean_feature_names
    import os
    from utils import *
    import openml
    from ft_detection import clean_feature_names
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'wine'
    for benchmark in ['TabArena', "Grinsztajn", "TabZilla"]:
        exp_name = f"EXP_cat_freq_{benchmark}"
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
            # if dataset_name not in data.name:
            #     continue
        
            
            if data.name in results['performance']:
                print(f"Skipping {data.name} as it already exists in results.")
                print(pd.DataFrame(results['performance'][data.name]).apply(lambda x: np.mean(np.mean(x)),axis=1))
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
            
            detector = CatFreqDetector(
                target_type=target_type, 
            )

            detector.fit(X, y)
            if detector.detection_attempted:
                results['performance'][data.name] = detector.scores
                results['significances'][data.name] = detector.significances
                
                print(pd.DataFrame(results['performance'][data.name]).apply(lambda x: np.mean(np.mean(x)),axis=1).sort_values())
                print()
        with open(f"{exp_name}.pkl", "wb") as f:
            pickle.dump(results, f)

    