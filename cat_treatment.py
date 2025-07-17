from statistics import LinearRegression
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sympy import Ordinal, rem
from proxy_models import TargetMeanClassifier, TargetMeanRegressor, TargetMeanClassifierCut, TargetMeanRegressorCut, UnivariateLinearRegressor, UnivariateLogisticClassifier
import openml
import pandas as pd
from utils import get_benchmark_dataIDs, get_metadata_df, make_cv_function
from ft_detection import FeatureTypeDetector

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from category_encoders import LeaveOneOutEncoder, OneHotEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from base_preprocessor import BasePreprocessor
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
                 **kwargs):
        super().__init__(target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, 
                         mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.min_cardinality = min_cardinality # TODO: Think about what to do with low-cardinality features


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

    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        # TODO
        pass


    def fit(self, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()

        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        
        cat_cols = [col for col in cat_cols if X[col].nunique() >= self.min_cardinality]
        if len(cat_cols) == 0:
            self.detection_attempted = False
            return self
        self.scores = {col: {} for col in cat_cols}
        self.significances = {col: {} for col in cat_cols}
        # 1. Get the dummy and target mean scores for the categorical features
        self.get_dummy_mean_scores(X[cat_cols], y)
        
        for col in cat_cols:       
            # IDEAS:
            # - Drop a feature if dummy>all conditions
            # - Use Ordinal encoding (or original values or sorted) if 'mean' is not significantly better     
            # - Need to think deeply about when and why OHE is better or worse than mean
                # Likely, if it is better, this might mean that it effectively avoids making overly confident predictions on unseen values by adding the constant
                # If it is worse, this might mean that we have a very predictive mean for the feature and we should use target encoding over OHE
            # - We could also try to define a percentage starting from which we use ordinal encoding, i.e. if the linear model is mot worse than x%
                # Another approach would be to use an absolute value, depending on how high the highest performance is
            # - BUT: It might as well make sense to also include the other interpolation methods as well as a combination test before deciding.
            
            
            
            # 2. Get the linear-float model
            if pd.to_numeric(X[col].dropna(), errors='coerce').notna().all():
                X_float = pd.to_numeric(X[col], errors='coerce')
                self.scores[col]['linear'] = self.single_interpolation_test(X_float, y, interpolation_method='linear')

            # 3. Get the ordinal model
            # TODO: Remove leak
            X_ordinal = pd.Series(OrdinalEncoder().fit_transform(X[[col]]).flatten(), name=col)
            self.scores[col]['linear-ordinal'] = self.single_interpolation_test(X_ordinal, y, interpolation_method='linear')

            # 4. Get the sorted model
            u_sorted = X[col].sort_values().unique()
            sorted_map = dict(zip(u_sorted, range(len(u_sorted))))
            X_sorted = X[col].map(sorted_map)
            self.scores[col]['linear-sorted'] = self.single_interpolation_test(X_sorted, y, interpolation_method='linear')

            # 5. Get the OHE model
            # X_ohe = pd.get_dummies(X[col], prefix=col, drop_first=True)
            self.scores[col]['ohe'] = self.cv_func(X[[col]].copy(), y,
                                                   Pipeline([
                                                       ("ohe",  OneHotEncoder(handle_unknown='value', handle_missing='indicator')),
                                                       ("model", self.linear_model)
                                                   ]))
            print(col, pd.DataFrame(self.scores[col]).mean().sort_values())
            print()
        self.detection_attempted = True
        # Filter candidates by distinctiveness
        # candidate_cols = self.filter_candidates_by_distinctiveness(X[cat_cols])

        # if len(candidate_cols) > 0:
        #     self.multivariate_performance_test(X, y_input, candidate_cols, suffix="ADDFREQ")
        #     self.detection_attempted = True
        # else:
        #     self.detection_attempted = False

        return self

    # def transform(self, X):
    #     if len(self.irrelevant_features)==0:
    #         return X

    #     return X.drop(self.irrelevant_features, axis=1)

if __name__ == "__main__":
    from ft_detection import clean_feature_names
    import os
    from utils import *
    import openml
    from ft_detection import clean_feature_names
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'wine'
    for benchmark in ['TabArena', "Grinsztajn", "TabZilla"]:
        exp_name = f"EXP_cat_treat_{benchmark}"
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
            
            detector = CatTreatmentDetector(
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

    