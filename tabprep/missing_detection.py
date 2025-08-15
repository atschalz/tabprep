import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from base_preprocessor import BasePreprocessor
class MissingDetector(BasePreprocessor):
    def __init__(self, 
                 target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                 n_values=4,
                 min_cardinality=6,
                 min_freq=2,
                 ):
        super().__init__(target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, 
                         mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        
        self.n_values = n_values
        self.min_cardinality = min_cardinality
        self.min_freq = min_freq
    
    def preprocess_irregular_negatives(self, x):
        x = x.copy()
        neg_values = x[x < 0].unique()
        new_x = x.where(x >= 0, np.nan)
        add_feats = [x.where(x == i, np.nan).isna().astype(int).rename(f"{x.name}_separate_{i}") for i in neg_values]
        return new_x, add_feats

    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
                # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
            if mode == 'backward': 
                pass
            elif mode == 'forward':
                for col in test_cols:
                    new_x, add_feats = self.preprocess_irregular_negatives(X_out[col])
                    X_out[col] = new_x
                    for feat in add_feats:
                        X_out[feat.name] = feat
        else:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                # TODO: Implement more efficiently
                for col_use in test_cols:
                    if col_use==col:
                        continue
                    new_x, add_feats = self.preprocess_irregular_negatives(X_out[col_use])
                    X_out[col_use] = new_x
                    for feat in add_feats:
                        X_out[feat.name] = feat
            elif mode == 'forward':
                new_x, add_feats = self.preprocess_irregular_negatives(X_out[col])
                X_out[col] = new_x
                for feat in add_feats:
                    X_out[feat.name] = feat
            
        return X_out

    def fit(self, X_in, y_in=None):
        '''
        Cases detected:
        - bank-marketing: pdays
        - credit_card_clients_default: pay features have -1 and -2 (seems fine though)
        - heloc: All features have negative codes from -1 to -3 
        - polish companies: Three cases, nut seems not to be a case of missing values
        - students_dropout_and_academic_success: inflation_rate is found, although it is fine

        - visualizing_soil: easting is found
        - albert (Grin): V2
        - default-of-credit-card-clients: Six candidates, likely same as in credit_card_clients_default
        - bank-marketing: 'V14'
        - MiniBooNE: 35 candidates
        - heloc: 22 candidates

        - artificial_characters: One candidate
        - MiniBooNE: 35 candidates
        - albert: 1 candidate
        
        Filtered cases:
        - kddcup09_appetency: One feature, but would likely be false detection as it occurs only once

        - Other observations:
        - There is not a single opposite case where a positive value is isolated

        '''
        X = X_in.copy()
        y = y_in.copy() if y_in is not None else None
        X_num = X.select_dtypes(include=[np.number])

        self.original_missing_cnt = {col: X_num[col].isnull().sum() for col in X_num.columns}

        # Try to detect all columns where the data is strictly positive expect for one value
        self.candidates = []
        for col in X_num.columns:
            if X_num[col].nunique() < self.min_cardinality:
                continue
            n_negative = sum(np.sign(X[col].unique())==-1)
            if n_negative == 0:
                continue 
            elif sum(np.sign(X[col])==-1)<self.min_freq:
                continue
        
            elif n_negative < self.n_values:
                self.candidates.append(col)
        if len(self.candidates) == 0:
            return self

        reassign_cols = self.multivariate_performance_test(X, y, self.candidates, suffix='irregular')
        print(f"Detected {len(self.candidates)} candidates for missing values: {self.candidates}")
            # for col in self.candidates:
            #    print(col, X[col].sort_values().unique())



if __name__ == "__main__":
    from ft_detection import clean_feature_names
    import os
    from utils import *
    import openml
    from ft_detection import clean_feature_names
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'wine'
    for benchmark in ['TabArena',"Grinsztajn", "TabZilla"]:
        exp_name = f"EXP_num_interaction_{benchmark}"
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
            
            detector = MissingDetector(
                target_type=target_type, 
            )

            detector.fit(X, y)

            results['performance'][data.name] = detector.scores
            if len(detector.candidates) > 0:
                print("!!!", data.name, len(detector.candidates))
