import numpy as np
import pandas as pd
from regex import D
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from fe_utils import fe_combine
from utils import sample_from_set
from itertools import combinations 
from utils import make_cv_function, p_value_wilcoxon_greater_than_zero
from proxy_models import TargetMeanRegressor, TargetMeanClassifier
from itertools import product, combinations
from category_encoders import LeaveOneOutEncoder
from sklearn.metrics import roc_auc_score, log_loss, root_mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DuplicateCountAdder(BaseEstimator, TransformerMixin):
    """
    A transformer that counts duplicate samples in the training set
    and appends a new feature with those counts at transform time.
    """

    def __init__(self, feature_name: str = "duplicate_count"):
        self.feature_name = feature_name

    def fit(self, X_in, y=None):
        """
        Learn the duplicate‐count mapping from X.
        
        Parameters
        ----------
        X : array‐like or DataFrame, shape (n_samples, n_features)
            Training data.

        y : Ignored.
        
        Returns
        -------
        self
        """
        X = X_in.copy()
        # If DataFrame, extract values but remember columns
        if isinstance(X, pd.DataFrame):
            self._is_df = True
            self._columns_ = X.columns.tolist()
            data = X.values
        else:
            self._is_df = False
            data = np.asarray(X)
        
        # Convert each row into a hashable tuple and count
        rows = [tuple(row) for row in data]
        self.counts_ = Counter(rows)
        return self

    def transform(self, X_in):
        """
        Append the duplicate‐count feature to X.
        
        Parameters
        ----------
        X : array‐like or DataFrame, shape (n_samples, n_features)
            New data to transform.
        
        Returns
        -------
        X_out : same type as X but with one extra column
        """
        X = X_in.copy()
        # Prepare raw array and remember if DataFrame
        if isinstance(X, pd.DataFrame):
            cols = X.columns.tolist()
            data = X.values
        else:
            data = np.asarray(X)
        
        # Look up counts (0 if unseen)
        new_feat = pd.Series([self.counts_.get(tuple(row), 0) for row in data], index=X.index, name=self.feature_name)
        
        out = pd.concat([X, new_feat], axis=1)
        return out

def make_cv_function_for_duplicated_data(target_type, n_folds=5, early_stopping_rounds=20, 
                    random_state=42, 
                    vectorized=False, verbose=False,
                    groups=None,
                    drop_modes=0,
                                       ):
    """CV creation function for vectorized versions of the TargetEncoderModels."""
    if target_type=='binary':
        scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif target_type=='regression':
        scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:   
        raise ValueError("target_type must be 'binary' or 'regression'")

    def cv_func(X_df_in, y_s_in, pipeline, return_iterations=False, return_preds=False, scale_y=False, 
                weight_strategy=None, add_cnt=False, add_loo=False, add_feat_cnt=False):
        X_df = X_df_in.copy()
        y_s = y_s_in.copy()
        scores = []
        iterations = []
        all_preds = []
        for train_idx, test_idx in cv.split(X_df, y_s, groups=groups):
            X_tr, y_tr = X_df.iloc[train_idx], y_s.iloc[train_idx]
            X_te, y_te = X_df.iloc[test_idx], y_s.iloc[test_idx]
            
            if scale_y:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                y_tr = pd.Series(scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel(), name=y_tr.name, index=y_tr.index)
                y_te = pd.Series(scaler.transform(y_te.values.reshape(-1, 1)).ravel(), name=y_te.name, index=y_te.index)
            
            X_tr_use = X_tr.copy()
            X_te_use = X_te.copy()
            
            if add_cnt:
                dh = DuplicateCountAdder()
                X_tr_use = dh.fit_transform(X_tr_use)
                X_te_use = dh.transform(X_te_use)
            if add_feat_cnt:
                for col in X_tr_use.columns:
                    cnt_map = X_tr_use[col].value_counts().to_dict()
                    X_tr_use[f'{col}_cnt'] = X_tr_use[col].map(cnt_map)
                    X_te_use[f'{col}_cnt'] = X_te_use[col].map(cnt_map)
                X_tr_use = X_tr_use.copy()
                X_te_use = X_te_use.copy()
            if add_loo:
                loo = LeaveOneOutEncoder()
                train_str = X_tr_use.astype(str).sum(axis=1)
                test_str = X_te_use.astype(str).sum(axis=1)
                X_tr_use['LOO'] = loo.fit_transform(train_str, y_tr)
                X_te_use['LOO'] = loo.transform(test_str)

            if weight_strategy in ['basic', 'reverse']:
                train_str = X_tr_use.astype(str).sum(axis=1)
                cnt_map = dict(train_str.value_counts())
                
                X_tr_use = X_tr_use.drop_duplicates()
                y_tr_use = y_tr.loc[X_tr_use.index]
                sample_weights = X_tr_use.astype(str).sum(axis=1).map(cnt_map).values
                if target_type == "regression":
                    y_tr_map = y_tr.groupby(train_str).mean().astype(float)
                elif target_type == "binary":
                    y_tr_map = y_tr.groupby(train_str).mean().round()
                y_tr_use = X_tr_use.astype(str).sum(axis=1).map(y_tr_map).values
                if weight_strategy == 'reverse':
                    sample_weights = 1 / sample_weights
            else:
                sample_weights = np.ones_like(y_tr)
                y_tr_use = y_tr

            final_model = pipeline.named_steps["model"]

            # if it's an LGBM model, pass in eval_set + callbacks
            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                pipeline.fit(
                    X_tr_use, y_tr_use,
                    **{
                        "model__eval_set": [(X_te_use, y_te)],
                        "model__callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                        "model__sample_weight": sample_weights,
                        # "model__verbose": False
                    }
                )
            else:
                # dummy / other estimators: plain fit
                pipeline.fit(X_tr, y_tr)

            # predict
            if target_type == "binary" and hasattr(pipeline, "predict_proba"):
                if vectorized:
                    preds = pipeline.predict_proba(X_te_use)[:, :, 1]
                else:
                    preds = pipeline.predict_proba(X_te_use)[:, 1]
            else:
                preds = pipeline.predict(X_te_use)

            if scale_y:
                # inverse transform predictions if scaled
                preds = pd.Series(scaler.inverse_transform(preds.reshape(-1, 1)).ravel(), name=y_te.name, index=y_te.index)
                y_tr = pd.Series(scaler.inverse_transform(y_tr.values.reshape(-1, 1)).ravel(), name=y_tr.name, index=y_tr.index)
                y_te = pd.Series(scaler.inverse_transform(y_te.values.reshape(-1, 1)).ravel(), name=y_te.name, index=y_te.index)

            if vectorized:
                scores.append({col: scorer(y_te, preds[:,num]) for num, col in enumerate(X_df.columns)})
            else:
                scores.append(scorer(y_te, preds))
            
            if return_preds:
                all_preds.append(pd.Series(preds, name=y_te.name, index=y_te.index))

            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                iterations.append(pipeline.named_steps['model'].booster_.num_trees())

        # TODO: Might change to return a dict instead
        if return_iterations and return_preds:
            return np.array(scores), iterations, all_preds
        elif return_iterations and not return_preds:
            return np.array(scores), iterations
        elif return_preds and not return_iterations:
            return np.array(scores), all_preds
        else:
            return np.array(scores)

    return cv_func

from base_preprocessor import BasePreprocessor
class DuplicateDetector(BasePreprocessor):
    def __init__(self, 
                 target_type, 
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                 threshold=0.05, 
                 only_with_matching_target=True,
                 handle_mode = 'postprocess', # ['postprocess', 'preprocess', 'warn']
                 
                 ):
        # TODO: Implement mode where we also try to detect duplicates using subsets of features
        # TODO: Test idea of using LOO encoder as proxy for the pd.Series of the duplicate ids as cat feature
        # TODO: Think about whether AUC as a rankling-based metric is even appropriate in the presence of duplicates
        super().__init__(target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, 
                         mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.threshold = threshold
        self.only_with_matching_target = only_with_matching_target
        self.handle_mode = handle_mode

        if self.target_type=='regression':
            self.metric = lambda y_true, y_pred: -root_mean_squared_error(y_true, y_pred)
            self.lgb_model = lgb.LGBMRegressor
        elif self.target_type=='binary':
            self.metric = lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
            self.lgb_model = lgb.LGBMClassifier

        self.cv_func = make_cv_function_for_duplicated_data(
            target_type=self.target_type, n_folds=self.n_folds)

    def dedupe_with_weights(df: pd.DataFrame,
                            subset: list[str] | None = None,
                            weight_col: str = "sample_weight") -> pd.DataFrame:
        """
        Remove duplicate rows from a DataFrame but keep track of how many times
        each unique row appeared via a sample-weight column.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame, possibly with duplicates.
        subset : list[str] | None, default None
            List of column names to consider when identifying duplicates.
            If None, all columns are used.
        weight_col : str, default "sample_weight"
            Name of the column to create that will hold the occurrence counts.

        Returns
        -------
        pd.DataFrame
            A deduplicated DataFrame with an extra column giving the count
            of how many times each row appeared in the original DataFrame.
        """
        # 1) Compute counts of each unique key
        counts = (
            df
            .groupby(subset or df.columns.tolist(), dropna=False)
            .size()
            .reset_index(name=weight_col)
        )

        # 2) Drop duplicates from the original DataFrame
        deduped = df.drop_duplicates(subset=subset, keep="first").copy()

        # 3) Merge the counts back onto the deduplicated rows
        result = deduped.merge(counts, how="left", on=subset or df.columns.tolist())

        return result

    def adjust_predictions(self, X: pd.DataFrame, y: pd.Series, preds: list[pd.Series],
                            method = 'mean', # ['mean', 'frequent', 'uniform', 'frequent_uniform']
                            ):
        adjusted_fold_perf = []
        for fold in range(self.n_folds):
            pred = preds[fold]
            X_test = X.loc[pred.index]
            X_train = X.drop(index=pred.index)
            y_test = y.loc[pred.index]
            y_train = y.drop(index=pred.index)
            
            train_str = X_train.astype(str).sum(axis=1)
            test_str = X_test.astype(str).sum(axis=1)
            
            df_count_mean = pd.DataFrame([train_str.value_counts(), y_train.groupby(train_str).mean().loc[train_str.value_counts().index]]).transpose()
            df_count_mean.columns = ['count', 'mean']
            
            if method == 'mean':
                dupe_maps = dict(df_count_mean.loc[:, 'mean'])
            elif method == 'frequent':
                dupe_maps = dict(df_count_mean.loc[df_count_mean['count']>1,'mean'])
            elif method == 'uniform':
                dupe_maps = dict(df_count_mean.loc[df_count_mean['mean'].apply(lambda x: x in [0,1]),'mean'])
            elif method == 'frequent_uniform':
                dupe_maps = dict(df_count_mean.loc[(df_count_mean['count']>1) & (df_count_mean['mean'].apply(lambda x: x in [0,1])), 'mean'])

            # new_pred = [float(dupe_maps[i])*float(pred.iloc[num]) if i in dupe_maps else float(pred.iloc[num])  for num, i in enumerate(test_str)]
            new_pred = [float(dupe_maps[i]) if i in dupe_maps else float(pred.iloc[num])  for num, i in enumerate(test_str)]
            new_pred = pd.Series(new_pred, index=pred.index, name=pred.name)

            adjusted_fold_perf.append(self.metric(y_test, new_pred))

        return np.array(adjusted_fold_perf)

    def fit(self, X_in, y_in):
        '''
        Some thoughts:
        - If we have duplicates in X, we have the following options:
            - Warn the user and let them decide how to handle it
            - Preprocess the data by removing duplicates and adding sample weights
            - Postprocess the data by assigning the expected value of the target to values seen during training
            - Other kinds of preprocessing (tbd)
            - Use that as a hint for other dataset properties (although those other properties are yet unclear)
        - While for images duplicates with different labels indicate label noise or mislabeling, for tabular data this should be perfectly fine, as there are always unobserved variables that can determine the outcome.

        '''

        X = X_in.copy()
        y = y_in.copy()

        self.n_X_duplicates = X.duplicated().mean()
        self.n_Xy_duplicates = pd.concat([X,y],axis=1).duplicated().mean()

        if self.n_X_duplicates < self.threshold:
            if self.n_X_duplicates == 0:
                print("No duplicates found in X.")
            else:
                print(f"Found {self.n_X_duplicates:.2%} duplicates in X. Ignore them as that is below threshold={self.threshold:.2%}.")
            return self
        
        if self.handle_mode == 'warn':
                print(f"Warning: {self.n_X_duplicates:.2%} of the samples in X are duplicates. This requires careful handling, as it may lead to overfitting or biased model evaluation.")
                if self.n_Xy_duplicates == self.n_X_duplicates:
                    print(f"The duplicates also match on the target. \
                        If it is expected that train samples can appear at test time, consider setting handle_mode to 'postprocess' for avoiding to predict the known samples. \
                            If this is not the case, consider setting handle_mode to 'preprocess' for reducing the samples to unique ones and adding sample weights to the model.")
                else:
                    print(f"The duplicates do not match on the target.")
                print('In any case, it is recommended to clarify whether the duplicated samples are intended. Most likely, they are due to factors like repeated experiments, omitted variable bias or artifacts of the data collection process. It is recommended to think about the right treatment.')

        # elif self.handle_mode == 'preprocess':
        #     X = self.dedupe_with_weights(X, subset=X.columns.tolist())
        # elif self.handle_mode == 'postprocess':
            # if self.n_X_duplicates==self.n_Xy_duplicates:
            #     X_deduplicated = X.drop_duplicates()
            #     y_deduplicated = y.loc[X_deduplicated.index]
            #     self.dupe_maps = dict(zip(X_deduplicated.astype(str).sum(axis=1).values,y_deduplicated.values))

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
    dataset_name = 'wine'
    for benchmark in ['TabArena']: # ["Grinsztajn", "TabArena", "TabZilla"]:
        exp_name = f"EXP_duplicates_{benchmark}"
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
            
            detector = DuplicateDetector(
                target_type=target_type, 
            )

            detector.fit(X, y)

            results['performance'][data.name] = detector.scores
            results['significances'][data.name] = detector.significances

            print(pd.DataFrame(results['performance'][data.name]).apply(lambda x: np.mean(np.mean(x)),axis=1).sort_values())

        with open(f"{exp_name}.pkl", "wb") as f:
            pickle.dump(results, f)

    