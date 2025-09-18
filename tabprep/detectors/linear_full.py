import numpy as np
import pandas as pd
from itertools import combinations 
from tabprep.utils.modeling_utils import make_cv_function, clean_feature_names
from tabprep.proxy_models import TargetMeanRegressor, TargetMeanClassifier, UnivariateLinearRegressor, UnivariateLogisticClassifier
from itertools import combinations
from sklearn.linear_model import LinearRegression, LogisticRegression
import lightgbm as lgb

'''
Next steps:
- Move combination test to BasePreprocessor
- combination test as a method to assess the performance of interaction features
- Add interaction test to BasePreprocessor
'''

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

# class LinearModelFeatureAdder(BaseEstimator, TransformerMixin):
#     def __init__(self, model=None, target_type='regression'):
#         self.target_type = target_type
#         if model is not None and not isinstance(model, (LinearRegression, LogisticRegression)):
#             raise ValueError("Model must be an instance of LinearRegression or LogisticRegression.")
#         elif model is None:
#             self.model = LinearRegression() if target_type == 'regression' else LogisticRegression()
#         else:
#             self.model = model
    
#     def fit(self, X, y):
#         self.use_feats = X.select_dtypes(include=[np.number]).columns.tolist()
#         self.model_ = Pipeline([
#             # PowerTransformer removes a few more features to be numeric than standaradscaler, mostly the very imbalanced ones
#             # ("standardize", PowerTransformer(method='yeo-johnson', standardize=True, )),
#             ("standardize", QuantileTransformer(n_quantiles=np.min([1000,X.shape[0]]), random_state=42)),
#             ("impute", SimpleImputer(strategy="median")),
#             # ("standardize", StandardScaler()),
#             ("model", clone(self.model))
#         ])
        
#         self.model_.fit(X[self.use_feats], y)
#         return self

#     def transform(self, X):
#         X_out = X.copy()
#         if self.target_type == 'regression':
#             linear_preds = self.model_.predict(X[self.use_feats])
#         else:
#             linear_preds = self.model_.predict_proba(X[self.use_feats])[:, 1]
#         X_out['linear'] = linear_preds
#         return X_out

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd

class LinearModelFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, model=None, target_type='regression'):
        self.target_type = target_type
        if model is not None and not isinstance(model, (LinearRegression, LogisticRegression)):
            raise ValueError("Model must be an instance of LinearRegression or LogisticRegression.")
        elif model is None:
            self.model = LinearRegression() if target_type == 'regression' else LogisticRegression()
        else:
            self.model = model
    
    def fit(self, X, y):
        self.use_feats = X.select_dtypes(include=[np.number]).columns.tolist()
        self.model_ = Pipeline([
            # PowerTransformer removes a few more features to be numeric than standaradscaler, mostly the very imbalanced ones
            # ("standardize", PowerTransformer(method='yeo-johnson', standardize=True, )),
            ("standardize", QuantileTransformer(n_quantiles=np.min([1000,X.shape[0]]), random_state=42)),
            ("impute", SimpleImputer(strategy="median")),
            # ("standardize", StandardScaler()),
            ("model", clone(self.model))
        ])
        
        self.model_.fit(X[self.use_feats], y)
        return self

    def transform(self, X):
        X_out = X.copy()
        if self.target_type == 'regression':
            linear_preds = self.model_.predict(X[self.use_feats])
        else:
            linear_preds = self.model_.predict_proba(X[self.use_feats])[:, 1]
        X_out['linear'] = linear_preds
        return X_out

# class LinearModelFeatureAdder(BaseEstimator, TransformerMixin):
#     def __init__(self, model=None, target_type='regression', n_splits=5, random_state=42):
#         self.target_type = target_type
#         self.n_splits = n_splits
#         self.random_state = random_state
#         if model is not None and not isinstance(model, (LinearRegression, LogisticRegression)):
#             raise ValueError("Model must be an instance of LinearRegression or LogisticRegression.")
#         self.model = model or (LinearRegression() if target_type == 'regression' else LogisticRegression())
    
#         if self.target_type == 'regression':
#             self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
#         else:
#             self.kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)


#     def fit(self, X, y, return_X = False):
#         self.use_feats = X.select_dtypes(include=[np.number]).columns.tolist()
#         self.models_ = []
#         self.oof_preds_ = np.zeros(X.shape[0])
#         self.fitted = True
        
#         for train_idx, valid_idx in self.kf.split(X, y):
#             X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
#             y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]

#             pipeline = Pipeline([
#                 ("standardize", StandardScaler()),
#                 ("impute", SimpleImputer(strategy="median")),
#                 ("model", clone(self.model))
#             ])
#             pipeline.fit(X_train[self.use_feats], y_train)

#             if self.target_type == 'regression':
#                 self.oof_preds_[valid_idx] = pipeline.predict(X_valid[self.use_feats])
#             else:
#                 self.oof_preds_[valid_idx] = pipeline.predict_proba(X_valid[self.use_feats])[:, 1]

#             self.models_.append(pipeline)
        
#         if return_X:
#             X_out = X.copy()
#             X_out['linear'] = self.oof_preds_
#             return self.transform(X)
#         else:
#             return self

#     def transform(self, X):
#         X_out = X.copy()

#         if hasattr(self, "fitted") and X.shape[0] == self.oof_preds_.shape[0]:
#             # Training data: use out-of-fold predictions
#             X_out['linear'] = self.oof_preds_
#         else:
#             # Test or new data: average prediction over all folds
#             preds = np.zeros(X.shape[0])
#             for model in self.models_:
#                 if self.target_type == 'regression':
#                     preds += model.predict(X[self.use_feats])
#                 else:
#                     preds += model.predict_proba(X[self.use_feats])[:, 1]
#             preds /= len(self.models_)
#             X_out['linear'] = preds

#         return X_out
    
#     def fit_transform(self, X, y = None, **fit_params):
#         return self.fit(X, y, return_X=True)



from tabprep.detectors.base_preprocessor import BasePreprocessor
class LinearTrendDetector(BasePreprocessor):
    def __init__(self, 
                 target_type, 
                n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                min_cardinality=6,
                combination_criterion='win',
                combination_test_min_bins=2,
                combination_test_max_bins=2048, #TODO: Test with 90% of unique values as max
                assign_numeric_with_combination=False,
                binning_strategy='lgb', # ['lgb', 'KMeans', 'DT']
                lgb_model_type="default"  # ["default", "fine_granular", "unique-based", "huge-capacity"]

                 ):
        # TODO: Include possibility to select operators
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.min_cardinality = min_cardinality
        self.combination_criterion = combination_criterion
        self.combination_test_min_bins = combination_test_min_bins
        self.combination_test_max_bins = combination_test_max_bins
        self.assign_numeric_with_combination = assign_numeric_with_combination
        self.binning_strategy = binning_strategy
        self.lgb_model_type = lgb_model_type

        self.linear_features = {}

        # self.get_linear_model = lambda: Pipeline([
        #     # PowerTransformer removes a few more features to be numeric than standaradscaler, mostly the very imbalanced ones
        #     # ("standardize", PowerTransformer(method='yeo-johnson', standardize=True, )),
        #     ("standardize", QuantileTransformer(n_quantiles=np.min([1000, int(x.shape[0]*(1-(1/self.n_folds)))]), random_state=42)),
        #     ("impute", SimpleImputer(strategy="median")),
        #     # ("standardize", StandardScaler()),
        #     ("model", LinearRegression() if self.target_type == 'regression' else LogisticRegression())
        # ])

    
    # def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
    #     if col is None:
    #         X_out = X_cand_in.copy()
    #         # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
    #             # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
    #         if mode == 'backward': 
    #             pass
    #         elif mode == 'forward':
    #             for col in test_cols:
    #                     if self.target_type=='regression':
    #                         X_out[col+'_linearized'] = self.linear_features[col].predict(X_out[col].to_frame())
    #                     else:
    #                         X_out[col+'_linearized'] = self.linear_features[col].predict_proba(X_out[col].to_frame())[:,1]
    #     else:
    #         X_out = X_cand_in.copy()
    #         if mode == 'backward':
    #             for use_col in test_cols:
    #                 if col != use_col:
    #                     if self.target_type=='regression':
    #                         X_out[use_col+'_linearized'] = self.linear_features[use_col].predict(X_out[use_col].to_frame())
    #                     else:
    #                         X_out[use_col+'_linearized'] = self.linear_features[use_col].predict_proba(X_out[use_col].to_frame())[:,1]
    #         elif mode == 'forward':
    #             for use_col in test_cols:
    #                 if col == use_col:
    #                     if self.target_type=='regression':
    #                         X_out[use_col+'_linearized'] = self.linear_features[use_col].predict(X_out[use_col].to_frame())
    #                     else:
    #                         X_out[use_col+'_linearized'] = self.linear_features[use_col].predict_proba(X_out[use_col].to_frame())[:,1]

    #     return X_out

    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        y = self.adjust_target_format(y)

        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(exclude=[np.number])

        if X_num.shape[1] == 0:
            return self
        
        prefix = "LGB-full"

        # TODO: Change to get_lgb_params
        params = self.adapt_lgb_params(X[X.nunique().idxmax()], )

        self.scores[prefix] = {}
        self.significances[prefix] = {}
        # Get performance for all base columns
        X_use = X.copy()
        X_use = X_use 
        obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
        X_use[obj_cols] = X_use[obj_cols].astype('category')
        model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        pipe = Pipeline([("model", model)])
        self.scores[prefix]['raw'] = self.cv_func(clean_feature_names(X_use), y, pipe, return_iterations=False)['scores']
        
        model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        pipe = Pipeline([
            # ("linear_add", LinearModelFeatureAdder(target_type=self.target_type)),
            ("model", model)
        ])
        
        self.scores[prefix][f"withLinear"] = self.cv_func(clean_feature_names(X_use), y, pipe, custom_prep=[LinearModelFeatureAdder(target_type=self.target_type)])['scores']

        self.significances[prefix][f"withLinear"] = self.significance_test(self.scores[prefix][f"withLinear"]-self.scores[prefix]['raw'])
        print("!")
        # self.linear_model = Pipeline([
        #     # PowerTransformer removes a few more features to be numeric than standaradscaler, mostly the very imbalanced ones
        #     # ("standardize", PowerTransformer(method='yeo-johnson', standardize=True, )),
        #     ("standardize", QuantileTransformer(n_quantiles=np.min([1000,X.shape[0]]), random_state=42)),
        #     ("impute", SimpleImputer(strategy="median")),
        #     # ("standardize", StandardScaler()),
        #     ("model", LinearRegression() if self.target_type == 'regression' else LogisticRegression())
        # ])


        # self.scores['linear'] = self.cv_func(X_num, y, self.linear_model)

        # self.linear_model.fit(X_num, y)

        # if self.target_type == 'regression':
        #     X['LINEAR'] = self.linear_model.predict(X_num)
        # else:
        #     X['LINEAR'] = self.linear_model.predict_proba(X_num)[:,1]

        # self.get_dummy_mean_scores(X_num[candidate_cols], y)
        # # TODO: Add filter to compare single results to LGB on the full dataset
        # # TODO: Test logic to start with the column that is strongest for linear and try only for that one
        # for cnum, col in enumerate(candidate_cols):         
        #     # TODO: Add detector for target encoding to be used whenever we have a very strong signal for that (single-feature-mean>lgb)   
        #     if self.verbose:
        #         print(f"\r{cnum}/{len(candidate_cols)} columns processed", end="", flush=True)
        #     x_use = X_num[col].copy()
        #     self.scores[col]['linear'] = self.single_interpolation_test(x_use, y, interpolation_method='linear')

        #     self.significances[col][f"significant_linear"] = self.significance_test(
        #                 self.scores[col]['linear'] - self.scores[col]["dummy"]
        #             )

        #     if self.significances[col][f"significant_linear"] > self.alpha:
        #         continue

        #     self.significances[col][f"test_linear_superior_mean"] = self.significance_test(
        #                 self.scores[col]['linear'] - self.scores[col]["mean"]
        #             ) 
        #     if self.significances[col][f"test_linear_superior_mean"] > self.alpha:
        #         continue
        #     # else:
        #     #     self.linear_features[col] = self.get_linear_model(x_use).fit(x_use.to_frame(), y)

        #     any_comb_superior = False
        #     m_bins = [2**i for i in range(1,100) if 2**i>= self.combination_test_min_bins and 2**i <= self.combination_test_max_bins]
        #     for m_bin in m_bins:
        #         ### Combination test
        #         nunique = x_use.nunique()
        #         if m_bin > nunique:
        #             continue
        #         # if nunique>10000:
        #         #     m_bin = int(10000*q)
        #         # else:
        #         #     m_bin = int(nunique*q)

        #         self.scores[col][f"combination_test_{m_bin}"] = self.single_combination_test(x_use, y, max_bin=m_bin, binning_strategy=self.binning_strategy)
        #         self.significances[col][f"test_linear_superior_combination_{m_bin}"] = self.significance_test(
        #                 self.scores[col]['linear'] - self.scores[col][f"combination_test_{m_bin}"]
        #             ) 
        #         if self.significances[col][f"test_linear_superior_combination_{m_bin}"] > self.alpha:
        #             any_comb_superior = True
        #             break
        #     if not any_comb_superior:
        #         self.linear_features[col] = self.get_linear_model(x_use).fit(x_use.to_frame(), y)


        # if len(self.linear_features)>0:
        # X_int = self.multivariate_performance_test(X, y,
        #                                         test_cols=['LINEAR'], 
        #                                         suffix='addlinear',
        #                                         )
        print(pd.DataFrame({col: pd.DataFrame(self.scores[col]).mean().sort_values() for col in self.scores})[prefix])
        return self
    
    # def transform(self, X):
    #     X_out = X.copy()

    #     return X_out
