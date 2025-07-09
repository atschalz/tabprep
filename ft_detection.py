import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, log_loss
from utils import TargetMeanClassifier,TargetMeanRegressor,UnivariateLinearRegressor,UnivariateLogisticClassifier,PolynomialLogisticClassifier,PolynomialRegressor, p_value_wilcoxon_greater_than_zero, clean_series, UnivariateThresholdClassifier, MultiFeatureTargetMeanClassifier, MultiFeatureUnivariateLogisticClassifier, make_cv_scores_with_early_stopping, LightGBMBinner, p_value_sign_test_median_greater_than_zero
from sklearn.linear_model import LogisticRegression
import os
import time
from category_encoders import LeaveOneOutEncoder


def clean_feature_names(X_input: pd.DataFrame):
    X = X_input.copy()
    X.columns = [str(col).replace('[', '').replace(']', '').replace(':', '')
                                            .replace('<', '').replace('>', '')
                                            .replace('=', '').replace(',', '')
                                            .replace(' ', '_') for col in X.columns]
    
    return X


class FeatureTypeDetector(TransformerMixin, BaseEstimator):
    def __init__(self, target_type, min_q_as_num=6, n_folds=5, 
                 lgb_model_type="unique-based-binned", 
                 interpolation_criterion="win", # ['win', 'match']
                 assign_numeric=False, 
                 assign_numeric_with_combination=False,
                 detect_numeric_in_string=False, 
                 use_highest_corr_feature=False, num_corr_feats_use=0,
                 fit_cat_models=True,
                 use_performance_test=False,
                 drop_unique=False,
                 combination_test_min_bins=16,
                 combination_test_max_bins=2048,
                 alpha=0.05,
                 interaction_mode='high_corr', # ['high_corr', 'all' int]
                 use_interaction_test=True,
                 use_leave_one_out_test=False,
                 drop_modes=0,
                 significance_method='wilcoxon', # ['t-test', 'wilcoxon', 'median]
                 verbose=True
                 ):

        # Parameters
        self.target_type = 'binary' if target_type == 'multiclass' else target_type
        self.min_q_as_num = min_q_as_num
        self.n_folds = n_folds
        self.lgb_model_type = lgb_model_type
        self.interpolation_criterion = interpolation_criterion
        self.assign_numeric = assign_numeric
        self.assign_numeric_with_combination = assign_numeric_with_combination
        self.detect_numeric_in_string = detect_numeric_in_string
        self.use_highest_corr_feature = use_highest_corr_feature
        self.num_corr_feats_use = num_corr_feats_use
        self.fit_cat_models = fit_cat_models
        self.use_performance_test = use_performance_test
        self.combination_test_min_bins = combination_test_min_bins
        self.combination_test_max_bins = combination_test_max_bins
        self.drop_unique = drop_unique
        self.alpha = alpha
        self.interaction_mode = interaction_mode
        self.use_interaction_test = use_interaction_test
        self.use_leave_one_out_test = use_leave_one_out_test
        self.drop_modes = drop_modes
        self.significance_method = significance_method
        self.verbose = verbose

        # Functions
        self.cv_scores_with_early_stopping = make_cv_scores_with_early_stopping(target_type=self.target_type, early_stopping_rounds=20, vectorized=False, drop_modes=self.drop_modes)
        self.cv_scores_with_early_stopping_vec = make_cv_scores_with_early_stopping(target_type=self.target_type, early_stopping_rounds=20, vectorized=True, drop_modes=self.drop_modes)
        if self.significance_method == 'wilcoxon':
            self.significance_test = p_value_wilcoxon_greater_than_zero
        elif self.significance_method == 't-test':
            from scipy.stats import ttest_rel
            self.significance_test = lambda x,y: ttest_rel(x, y).pvalue
        elif self.significance_method == 'median':
            from scipy.stats import median_test
            self.significance_test = p_value_sign_test_median_greater_than_zero

        # Output variables
        self.orig_dtypes = {}
        self.dtypes = {}
        self.col_names = []
        self.reassigned_features = []
        self.cat_dtype_maps = {}
        self.scores = {}
        self.significances = {}
        # TODO: Fix multi-class behavior
        # TODO: Think again whether the current proxy model really is the best choice (Might use a small NN instead of LGBM; Might use different HPs )
        # TODO: Implement the parallelization speedup for all interpolation tests 
        # TODO: Track how many features have been filtered by which step (mainly needs adding this for interpolation and combination tests)
        # TODO: Implement option to drop duplicates
        # TODO: Separate repetitive functions like the interpolation test into a separate function
        # TODO: Add option to add an additional categorical feature instead of reassigning the feature type
        # TODO: Add option to treat low-cardinality features as categorical

    def handle_trivial_features(self, X_input: pd.DataFrame, verbose=False):
        X = X_input.copy()
        col_names = X.columns
        ### 1. Assign binary type (assume uniform columns have not been handled before and assign them to be binary)
        ### 2. Handle low-cardinality - (OHE? Leave as cat?)
        bin_cols = []
        lowcard_cols = []
        for col in col_names:
            if X[col].nunique()<=2:
                self.dtypes[col] = "binary"
                bin_cols.append(col)
            elif X[col].nunique()<self.min_q_as_num:
                self.dtypes[col] = "low-cardinality"
                lowcard_cols.append(col)
        
        if verbose:
            print(f"{len(bin_cols)}/{len(col_names)} columns are binary")   
        rem_cols = [x for x in col_names if x not in bin_cols]

        if len(bin_cols) == len(col_names):
            return list()
        if verbose:
            print(f"{len(lowcard_cols)}/{len(rem_cols)} columns are low-cardinality")        
        rem_cols = [x for x in rem_cols if x not in lowcard_cols]
        
        ### 3. Clearly categoricals
        num_coerced = np.array([int(pd.to_numeric(X[col].dropna(), errors= 'coerce').isna().sum()) for col in rem_cols])
        
        numeric_cand_cols = np.array(rem_cols)[num_coerced==0].tolist()
        if verbose:
            print(f"{len(numeric_cand_cols)}/{len(num_coerced)} columns can be converted to floats")        
        rem_cols = [x for x in rem_cols if x not in numeric_cand_cols]
        
        num_coerced = np.array([int(pd.to_numeric(X[col].dropna(), errors= 'coerce').isna().sum()) for col in rem_cols])
        if self.detect_numeric_in_string: # Currently doesn't work!!!! - see TODO below
            cat_cols = np.array(rem_cols)[[num_coerced[num]==X[col].dropna().shape[0] for num, col in enumerate(rem_cols)]].tolist()
        else:
            cat_cols = np.array(rem_cols)[[num_coerced[num]<=X[col].dropna().shape[0] for num, col in enumerate(rem_cols)]].tolist()
        
        if verbose:
            print(f"{len(cat_cols)}/{len(num_coerced)} columns are entirely categorical")        
        rem_cols = [x for x in rem_cols if x not in cat_cols]
        for col in cat_cols:
            self.dtypes[col] = "categorical"

        # 4. Try to extract numerical features from string columns
        # TODO: To apply this, we would need to integrate it in the fit-transform logic. Therefore, disable for now.
        # if len(rem_cols)>0:
        #     num_coerced = np.array([int(pd.to_numeric(X[col].dropna(), errors= 'coerce').isna().sum()) for col in rem_cols])
        #     part_coerced = np.array(rem_cols)[[0<c<X[col].dropna().shape[0] for c, col in zip(num_coerced, rem_cols)]].tolist()
        #     if len(part_coerced)>0:
        #         X_copy = X.loc[:, part_coerced].apply(clean_series)
        #         X[part_coerced] = X_copy
        #     all_nan = X_num.columns[X_num.isna().mean()==1]
        #     if verbose:
        #         print(f"{len(part_coerced)}/{len(rem_cols)} columns are partially numerical. {len(all_nan)} of them don't show regular patterns and are treated as categorical.")        
            
        #     if len(all_nan)>0:
        #         for col in all_nan:
        #             self.dtypes[col] = "categorical"
        #     rem_cols = [x for x in rem_cols if x not in part_coerced]
        #     numeric_cand_cols += [x for x in part_coerced if x not in all_nan]
        
        assert len(rem_cols)==0
        
        return numeric_cand_cols


    def get_dummy_mean_scores(self, X, y):
        X_use = X.copy()
        ### Get dummy and target mean scores
        irrelevant_cols = []
        for cnum, col in enumerate(X_use.columns):            
            x_use = X_use[col].copy()
            self.scores[col] = {}
            self.significances[col] = {}
            if self.verbose:
                print(f"\rDummy and target mean test: {cnum+1}/{len(X_use.columns)} columns processed", end="", flush=True)
            self.significances[col] = {}

            # dummy baseline on single column
            dummy_pipe = Pipeline([("model", DummyRegressor(strategy='mean') if self.target_type=="regression" else DummyClassifier(strategy='prior'))])
            self.scores[col]["dummy"] = self.cv_scores_with_early_stopping(x_use.to_frame(), y, dummy_pipe)
            
            model = (TargetMeanClassifier() if self.target_type=="binary"
                    else TargetMeanRegressor())
            pipe = Pipeline([("model", model)])
            self.scores[col]["mean"] = self.cv_scores_with_early_stopping(x_use.astype('category').to_frame(), y, pipe)

            # TODO: Consider adding unpredictive mean test - if the mean is not predictive at all, there is no categorical feature (Might as well be a bad idea as features might be uncorrelated to the target while still affecting other features effect on the target)
            self.significances[col]["test_irrelevant_mean"] = self.significance_test(
                self.scores[col]["mean"] - self.scores[col]["dummy"]
            )
            # TODO: Check what would be the right testing condition (something like "if mean not significantly better than dummy, then we can assume the feature is irrelevant")
            if self.significances[col]["test_irrelevant_mean"]>self.alpha: #'mean equal or worse than dummy'
                self.dtypes[col] = "irrelevant"
                irrelevant_cols.append(col)

        return [col for col in X_use.columns if col not in irrelevant_cols]

    def single_interpolation_test(
            self,
            x, y, 
            interpolation_method='linear',
    ):        
        if not x.dtype in ["int", "float", "bool"]:
            x = pd.to_numeric(x, errors='coerce')

        if interpolation_method == 'linear':
            interpol_model = UnivariateLogisticClassifier() if self.target_type == "binary" else UnivariateLinearRegressor()
        elif interpolation_method == 'poly2':
            interpol_model = PolynomialLogisticClassifier(degree=2) if self.target_type == "binary" else PolynomialRegressor(degree=2)
        elif interpolation_method == 'poly3':
            interpol_model = PolynomialLogisticClassifier(degree=3) if self.target_type == "binary" else PolynomialRegressor(degree=3)
        elif interpolation_method == 'poly4':
            interpol_model = PolynomialLogisticClassifier(degree=4) if self.target_type == "binary" else PolynomialRegressor(degree=4)
        elif interpolation_method == 'poly5':
            interpol_model = PolynomialLogisticClassifier(degree=5) if self.target_type == "binary" else PolynomialRegressor(degree=5)
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")

        pipe = Pipeline([
            # PowerTransformer removes a few more features to be numeric than standaradscaler, mostly the very imbalanced ones
            # ("standardize", PowerTransformer(method='yeo-johnson', standardize=True, )),
            ("standardize", QuantileTransformer(n_quantiles=np.min([1000, int(x.shape[0]*(1-(1/self.n_folds)))]), random_state=42)),
            ("impute", SimpleImputer(strategy="median")),
            # ("standardize", StandardScaler()),
            ("model", interpol_model)
        ])

        return self.cv_scores_with_early_stopping(x.to_frame(), y, pipe)

    def interpolation_test(self, X, y, max_degree=3, assign_dtypes=True):
        X_use = X.copy()
        remaining_cols = X_use.columns.tolist()

        for d in range(1, max_degree+1):
            if d==1:
                interpolation_method = 'linear'
            elif d==2:
                interpolation_method = 'poly2'
            elif d==3:
                interpolation_method = 'poly3'
            elif d==4:
                interpolation_method = 'poly4'
            elif d==5:
                interpolation_method = 'poly5'
            else:
                raise ValueError(f"Unsupported degree: {d}")
            
        
            ### Interpolation test
            interpol_cols = []
            for cnum, col in enumerate(X_use.columns):
                if cnum==0:
                    minus_done = 0
                if len(remaining_cols)==0:
                    break                
                if col not in remaining_cols: 
                    minus_done += 1
                    continue                
                if self.verbose:
                    print(f"\r{interpolation_method} interpolation test: {cnum-minus_done+1}/{len(remaining_cols)} columns processed", end="", flush=True)
                
                x_use = X_use[col].copy()
                self.scores[col][interpolation_method] = self.single_interpolation_test(x_use, y, interpolation_method=interpolation_method)
                
                self.significances[col][f"test_{interpolation_method}_superior"] = self.significance_test(
                        self.scores[col][interpolation_method] - self.scores[col]["mean"]
                    )

                self.significances[col][f"test_{interpolation_method}_superior_dummy"] = self.significance_test(
                        self.scores[col][interpolation_method] - self.scores[col]["dummy"]
                    )

                self.significances[col][f"test_mean_superior_to_{interpolation_method}"] = self.significance_test(
                         self.scores[col]["mean"] - self.scores[col][interpolation_method]
                    )

                # TODO: Consider adding a hyperparameter to accept a feature as numeric if linear matches the mean performance - in this case the pattern is so simple, that we don't need the complexity of treating a feature as categorical
                if self.interpolation_criterion == "win":
                    if self.significances[col][f"test_{interpolation_method}_superior"]<self.alpha:
                        if assign_dtypes:
                            self.dtypes[col] = "numeric"
                        interpol_cols.append(col)
                elif self.interpolation_criterion == "match":
                    if self.significances[col][f"test_mean_superior_to_{interpolation_method}"]>self.alpha:
                        if assign_dtypes:
                            self.dtypes[col] = "numeric"
                        interpol_cols.append(col)

            if self.verbose:
                print("\n")
                print(f"{len(interpol_cols)}/{len(remaining_cols)} columns are numeric acc. to {interpolation_method} interpolation test.")

            remaining_cols = [x for x in remaining_cols if x not in interpol_cols]

        return remaining_cols

    def single_combination_test(
            self,
            x, y, 
            max_bin=255,
    ):        
        if not x.dtype in ["int", "float", "bool"]:
            x = pd.to_numeric(x, errors='coerce')

        pipe = Pipeline([
            ("binning", LightGBMBinner(max_bin=max_bin)),
            ("model", TargetMeanClassifier() if self.target_type == "binary" else TargetMeanRegressor())
        ])

        return self.cv_scores_with_early_stopping(x.to_frame(), y, pipe)
    
    def combination_test(self, X, y, max_binning_configs=3, early_stopping=True, assign_dtypes = True):
        '''
        assign_dtypes was mainly added for the interaction tests
        '''
        X_use = X.copy()
        remaining_cols = X_use.columns.tolist()
        # for q in np.linspace(0.1,1,max_binning_configs, endpoint=False):
        m_bins = [2**i for i in range(1,100) if 2**i>= self.combination_test_min_bins and 2**i <= self.combination_test_max_bins]
        for m_bin in m_bins:
            ### Combination test
            comb_cols = []
            for cnum, col in enumerate(X_use.columns):
                if cnum==0:
                    minus_done = 0
                nunique = X_use[col].nunique()
                if m_bin > nunique:
                    continue
                # if nunique>10000:
                #     m_bin = int(10000*q)
                # else:
                #     m_bin = int(nunique*q)

                if early_stopping and len(remaining_cols)==0:
                    break                
                if early_stopping and col not in remaining_cols: # early_stopping=True is used if we just want to quickly find numeric features. If best performance matters, we would rather try all bins
                    minus_done += 1
                    continue                
                if self.verbose:
                    print(f"\rCombination test with max_bin of {m_bin}: {cnum-minus_done+1}/{len(remaining_cols)} columns processed", end="", flush=True)

                x_use = X_use[col].copy()
                self.scores[col][f"combination_test_{m_bin}"] = self.single_combination_test(x_use, y, max_bin=m_bin)
                self.significances[col][f"test_combination_test_{m_bin}_superior"] = self.significance_test(
                        self.scores[col][f"combination_test_{m_bin}"] - self.scores[col]["mean"]
                    )
                
                if self.significances[col][f"test_combination_test_{m_bin}_superior"]<self.alpha:
                    if assign_dtypes:
                        self.dtypes[col] = "numeric"
                    comb_cols.append(col)

            if self.verbose:
                print("\n")
                print(f"{len(comb_cols)}/{len(remaining_cols)} columns are numeric acc. to combination test with {m_bin} bins.")

            remaining_cols = [x for x in remaining_cols if x not in comb_cols]

        return remaining_cols

    def adapt_lgb_params(self, x: pd.Series, base_params: dict):
        params = base_params.copy()
        if self.lgb_model_type=="unique-based":
            params["max_bin"] = x.nunique() #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 2 #if mode=="cat" else 2
            params["n_estimators"] = x.nunique() # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif self.lgb_model_type=="huge-capacity":
            params["max_bin"] = int(x.nunique()) #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["min_data_in_leaf"] = 1 #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 5 #if mode=="cat" else 2
            params["n_estimators"] = 10000 # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif self.lgb_model_type=="full-capacity":
            params["max_bin"] = int(x.nunique()) #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["min_data_in_leaf"] = 1 #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 2 #if mode=="cat" else 2
            params["n_estimators"] = int(x.nunique()*2) # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif self.lgb_model_type=="unique-based-binned":
            params["max_bin"] = int(x.nunique()/2) #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 2 #if mode=="cat" else 2
            params["n_estimators"] = x.nunique() # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif self.lgb_model_type=="high-capacity":
            params["max_bin"] = x.nunique() #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif self.lgb_model_type=="high-capacity-binned":
            params["max_bin"] = int(x.nunique()/2) #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)
        else:
            raise ValueError(f"Unknown lgb_model_type: {self.lgb_model_type}. Use 'unique-based', 'high-capacity', 'high-capacity-binned', 'full-capacity' or 'default'.")
        
        return params
    
    def performance_test(self, X, y):
        base_params = {
            "objective": "binary" if self.target_type=="binary" else "regression",
            "boosting_type": "gbdt",
            # "n_estimators": 1000,
            "verbosity": -1
        }

        perf_cols = []
        for cnum, col in enumerate(X.columns):                
            print(f"\Performance test: {cnum+1}/{len(X.columns)} columns processed", end="", flush=True)
            x_use = X[[col]].copy()

            params = self.adapt_lgb_params(x_use[col], base_params.copy())

            model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
            pipe = Pipeline([("model", model)])
            self.scores[col]["lgb"] = self.cv_scores_with_early_stopping(clean_feature_names(x_use), y, pipe)

            self.significances[col]["test_lgb_superior"] = self.significance_test(
                self.scores[col]["lgb"] - self.scores[col]["mean"]
            )
            
            self.significances[col]["test_mean_superior_lgb"] = self.significance_test(
                self.scores[col]["mean"] - self.scores[col]["lgb"]
            )
            
            self.significances[col]["test_lgb_beats_dummy"] = self.significance_test(
                self.scores[col]["lgb"] - self.scores[col]["dummy"]
            )

            if self.significances[col][f"test_lgb_superior"]<self.alpha:
                self.dtypes[col] = "numeric"
                perf_cols.append(col)

        if self.verbose:
            print("\n")
            print(f"{len(perf_cols)}/{len(X.columns)} columns are numeric acc. to combination test with {m_bin} bins.")

        remaining_cols = [x for x in X.columns if x not in perf_cols]

        return remaining_cols

    def leave_one_out_test(self, X, y):
        if self.target_type in ['binary', 'multiclass']:
            dummy = DummyClassifier(strategy='prior')
            metric = roc_auc_score
        elif self.target_type == 'regression':
            dummy = DummyRegressor(strategy='mean')
            metric = lambda x,y: -root_mean_squared_error(x, y)

        # X_use = X.copy()
        loo_cols = []
        self.loo_scores = {}
        for cnum, col in enumerate(X.columns):                
            if self.verbose:
                print(f"\rLeave-One-Out test: {cnum+1}/{len(X.columns)} columns processed", end="", flush=True)
            dummy_pred = dummy.fit(X[[col]], y).predict(X[[col]])
            dummy_score = metric(y, dummy_pred)

            loo_pred = LeaveOneOutEncoder().fit_transform(X[col].astype('category'), y)[col]
            self.loo_scores[col] = metric(y, loo_pred)

            if self.loo_scores[col] < dummy_score:
                self.dtypes[col] = "numeric"
                loo_cols.append(col)

        if self.verbose:
            print("\n")
            print(f"{len(loo_cols)}/{len(X.columns)} columns are numeric acc. to Leave-One-Out test.")

        remaining_cols = [x for x in X.columns if x not in loo_cols]

        return remaining_cols
    
    def interaction_test(self, X_cand, X_full, y):
        '''Three aspects matter:
        1. Does the new feature behaves numerically? - are the interpolation and combination tests positive?
        2. Does the feature improve performance over the base features at all?
        3. Is the feature interaction truly numerical or just the same as a combination of two base features?
        Therefore, we label a feature as numerical if its interaction 
            a) behaves numerically,
            b) improves performance, and 
            c) is not just a combination of the base features. (TODO)
        '''
        # TODO: Might need to account for the fact that some datasets have many categorical and possibly just one numerical feature s.t. numerical interactions cannot be tested but the feature can still interact numerically with other categoricals
        if self.interaction_mode == 'corr_improve_over_base':
            base_cors = X_full.corrwith(y).abs()
            

        interaction_cols = []
        for col in X_cand.columns:
            
            ### 1. Get Interactions
            cols_use = X_full.drop(col,axis=1).columns

            X_int = pd.DataFrame(
                index=X_full.index, 
                columns=[f"{col}_/_{col2}" for col2 in cols_use]+ \
                [f"{col}_x_{col2}" for col2 in cols_use]+ \
                [f"{col}_-_{col2}" for col2 in cols_use]+ \
                [f"{col}_+_{col2}" for col2 in cols_use]
            )
            
            # TODO: Think about whether reversing - and / makes sense
            # TODO: Think about whether setting nans for division by zero is appropriate
            X_int[[f"{col}_/_{col2}" for col2 in cols_use]] = (X_full[[col]*len(cols_use)] / X_full[cols_use].replace(0, np.nan).values).values
            X_int[[f"{col}_x_{col2}" for col2 in cols_use]] = (X_full[[col]*len(cols_use)] * X_full[cols_use].values).values
            X_int[[f"{col}_-_{col2}" for col2 in cols_use]] = (X_full[[col]*len(cols_use)] - X_full[cols_use].values).values
            X_int[[f"{col}_+_{col2}" for col2 in cols_use]] = (X_full[[col]*len(cols_use)] + X_full[cols_use].values).values

            # Filter weird cases
            X_int = X_int.loc[:, X_int.nunique()>2]

            ### 2. Get Highest Correlation
            corr_X_int = X_int.corrwith(y, method='spearman').abs().sort_values(ascending=False)
            highest_corr = corr_X_int.index[0]
            if self.interaction_mode == 'high_corr':
                candidate_cols = [highest_corr]
            elif self.interaction_mode == 'all':
                candidate_cols = corr_X_int.index.tolist()
            elif self.interaction_mode == 'corr_improve_over_base':
                int_cors = X_int.corrwith(y, method='spearman').abs().to_frame()
                int_cors_diff = int_cors.apply(lambda x: float([x-base_cors.loc[x.name.split(f"_{i}_")].max() for i in ['+', '-', '/', 'x'] if len(x.name.split(f"_{i}_"))>1][0][0]),axis=1)
                candidate_cols = int_cors_diff.sort_values(ascending=False).index[:1].tolist()
            elif isinstance(self.interaction_mode, int):
                candidate_cols = corr_X_int.index[:self.interaction_mode].tolist()
            else:
                candidate_cols = []
                print(f"Unknown interaction_mode: {self.interaction_mode}. Use 'high_corr', 'all' or an integer.")                
            
            for highest_corr in candidate_cols:
                col2 = [highest_corr.split(f"_{i}_")[1] for i in ['+', '-', '/', 'x'] if len(highest_corr.split(f"_{i}_"))>1][0]
                X_use = X_int[[highest_corr]]
                arithmetic_col = X_use.columns[0]
                self.scores[arithmetic_col] = {}
                self.significances[arithmetic_col] = {}
                
                # base_performance = pd.Series({col: np.mean(self.scores[col]['mean']) for col in [col,col2]})
                # stronger_col = base_performance.idxmax()
                stronger_col, stronger_col_setting = pd.concat({col: pd.Series(self.scores[col]).apply(lambda x: x.mean()) for col in [col,col2]}).idxmax()

                ### 3. Get regular, binned and polynomial performance of the interaction feature and test significance
                # Regular TE performance
                self.get_dummy_mean_scores(X_use, y)
                self.significances[col][f"test_arithmetic-mean_superior_single-best"] = self.significance_test(
                    self.scores[arithmetic_col]['mean'] - self.scores[stronger_col][stronger_col_setting]
                )       
                if self.significances[col][f"test_arithmetic-mean_superior_single-best"]<self.alpha:
                    arithmetic_improves_performance = True
                else:
                    arithmetic_improves_performance = False

                ### Combination test
                # TODO: Make sure interpolation and combination tests for both features ran and comparing against the mean is appropriate (might need to compute additional scores for col2)
                self.combination_test(X_use, y, early_stopping=False, assign_dtypes=False)
                # 1. CHECK - The arithmetic combination feature behaves numerically
                arithmetic_is_numeric = False
                best_setting = "mean"
                m_bins = [2**i for i in range(1,100) if 2**i>= self.combination_test_min_bins and 2**i <= self.combination_test_max_bins]
                for m_bin in m_bins:
                    nunique = X_use[arithmetic_col].nunique()
                    if m_bin > nunique:
                        continue
                    if self.significances[arithmetic_col][f"test_combination_test_{m_bin}_superior"]<self.alpha:
                        arithmetic_is_numeric = True
                        best_setting = f"combination_test_{m_bin}"

                    self.significances[col][f"test_arithmetic-combination{m_bin}_superior_single-best"] = self.significance_test(
                        self.scores[arithmetic_col][f"combination_test_{m_bin}"] - self.scores[stronger_col][stronger_col_setting]
                    )       
                    if self.significances[col][f"test_arithmetic-combination{m_bin}_superior_single-best"]<self.alpha:
                        arithmetic_improves_performance = True

                    if arithmetic_is_numeric and arithmetic_improves_performance:
                        break
                
                # if not arithmetic_is_numeric:
                #     max_degree = 3 # TODO: Make this a hyperparameter
                #     self.interpolation_test(X_use, y, max_degree=max_degree, assign_dtypes=False)
                #     for d in range(1, max_degree+1):
                #         if d==1:
                #             interpolation_method = 'linear'
                #         elif d==2:
                #             interpolation_method = 'poly2'
                #         elif d==3:
                #             interpolation_method = 'poly3'
                #         elif d==4:
                #             interpolation_method = 'poly4'
                #         elif d==5:
                #             interpolation_method = 'poly5'
                #         else:
                #             raise ValueError(f"Unsupported degree: {d}")
                        
                #         if self.interpolation_criterion == "win":
                #             if self.significances[arithmetic_col][f"test_{interpolation_method}_superior"]<self.alpha:
                #                 arithmetic_is_numeric = True
                #                 best_setting = interpolation_method
                #                 break
                #         elif self.interpolation_criterion == "match":
                #             if self.significances[arithmetic_col][f"test_mean_superior_to_{interpolation_method}"]>self.alpha:
                #                 arithmetic_is_numeric = True
                #                 # best_setting = interpolation_method # TODO: unsure whether this is the right condition 
                #                 break

                # 2. CHECK - The arithmetic combination feature improves performance over the base features
                # TODO: Add significance level as a global hyperparameter
                self.significances[col][f"test_arithmetic-best_superior_single-best"] = self.significance_test(
                    self.scores[arithmetic_col][best_setting] - self.scores[stronger_col][stronger_col_setting]
                )       
                if self.significances[col][f"test_arithmetic-best_superior_single-best"]<self.alpha:
                    arithmetic_improves_performance = True

                # 3. CHECK - The arithmetic combination feature is not just a combination of the base features
                # TODO: Might implement that, but likely not needed

                if arithmetic_is_numeric and arithmetic_improves_performance:
                    self.dtypes[col] = "numeric"
                    interaction_cols.append(col)
                    break


                ### 5. Test whether categorical interaction is better than numerical interaction
                # Combine interaction performance
                # x = X_full[col].astype(str) + X_full[col2].astype(str)
                # if self.target_type == "regression":
                #     self.scores[arithmetic_col]['combine'] = self.cv_scores_with_early_stopping(x.to_frame(), y, Pipeline(steps=[('model', TargetMeanRegressor())]))
                # else:
                #     self.scores[arithmetic_col]['combine'] = self.cv_scores_with_early_stopping(x.to_frame(), y, Pipeline(steps=[('model', TargetMeanClassifier())]))
                

        remaining_cols = [x for x in X_cand.columns if x not in interaction_cols]

        return remaining_cols            

    def fit(self, X_input, y_input=None, verbose=False):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)
        if not isinstance(y_input, pd.DataFrame):
            y_input = pd.Series(y_input)
        
        self.orig_dtypes = {col: "categorical" if dtype in ["object", "category", str] else "numeric" for col,dtype in dict(X_input.dtypes).items()}
        for col in self.orig_dtypes:
            if X_input[col].nunique()<=2:
                self.orig_dtypes[col] = "binary"
        
        X = X_input.copy()
        y = y_input.copy()

        if self.target_type=="multiclass":
            # TODO: Fix this hack
            y = (y==y.value_counts().index[0]).astype(int)  # make it binary
            self.target_type = "binary"
        elif self.target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
            y = (y==y.value_counts().index[0]).astype(int)  # make it numeric

        self.col_names = X.columns
        self.dtypes = {col: "None" for col in  self.col_names}


        numeric_cand_cols: list[str] = self.handle_trivial_features(X, verbose=verbose) # type: ignore
        if len(numeric_cand_cols)==0:
            return

        if self.verbose:
            print(f"{len(numeric_cand_cols)} columns left for numeric/categorical detection")
        X_num = X[numeric_cand_cols].copy().astype(float)
        X_full = X_num.copy()

        ### Irrelevance test
        # Get dummy and target mean scores
        relevant_cols = self.get_dummy_mean_scores(X_num, y)
        if verbose:
            print(f"\r{len(relevant_cols)/len(numeric_cand_cols):.2%} of the numeric candidate columns are relevant according to the dummy and target mean test.")
        X_num = X_num[relevant_cols].copy()

        ### Leave-one-out test
        if self.use_leave_one_out_test:
            numeric_cand_cols = self.leave_one_out_test(X_num, y)
            X_num = X_num[numeric_cand_cols].copy()
            if len(numeric_cand_cols)==0:
                return

        ### Combination test
        numeric_cand_cols = self.combination_test(X_num, y)
        X_num = X_num[numeric_cand_cols].copy()
        if len(numeric_cand_cols)==0:
            return

        ### Linear interpolation test
        numeric_cand_cols = self.interpolation_test(X_num, y, max_degree=3)
        X_num = X_num[numeric_cand_cols].copy()
        if len(numeric_cand_cols)==0:
            return

        ### Interaction test
        if self.use_interaction_test:
            numeric_cand_cols = self.interaction_test(X_num, X_full, y)
            X_num = X_num[numeric_cand_cols].copy()
            if len(numeric_cand_cols)==0:
                return
        
        ### LightGBM performance test
        if self.use_performance_test:
            numeric_cand_cols = self.performance_test(X_num, y)
            X_num = X_num[numeric_cand_cols].copy()
            if len(numeric_cand_cols)==0:
                return

        for col in numeric_cand_cols:
            self.dtypes[col] = "categorical"

        # # TODO: Combination test likely is only required if there are still numeric candidate columns left that were not already classified as categorical by the previous tests
        # ### Combination test
        # # TODO: Test whether using a sklearn decision tree can speed things up
        # base_params = {
        #     "objective": "binary" if self.target_type=="binary" else "regression",
        #     "boosting_type": "gbdt",
        #     # "n_estimators": 1000,
        #     "verbosity": -1
        # }

        # comb_num = []
        # comb_cat = []
        # comb_bothfine = []
        # irrelevant_cols = []
        # for cnum, col in enumerate(numeric_cand_cols):                
        #     print(f"\rCombination test: {cnum+1}/{len(numeric_cand_cols)} columns processed", end="", flush=True)
        #     x_use = X_num[[col]].copy()

        #     params = self.adapt_lgb_params(x_use[col], base_params.copy())

        #     model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        #     pipe = Pipeline([("model", model)])
        #     self.scores[col]["lgb"] = self.cv_scores_with_early_stopping(clean_feature_names(x_use), y, pipe)


        #     self.significances[col]["test_lgb_superior"] = self.significance_test(
        #         self.scores[col]["lgb"] - self.scores[col]["mean"]
        #     )
            
        #     self.significances[col]["test_mean_superior"] = self.significance_test(
        #         self.scores[col]["mean"] - self.scores[col]["lgb"]
        #     )
            
        #     self.significances[col]["test_lgb_beats_dummy"] = self.significance_test(
        #         self.scores[col]["lgb"] - self.scores[col]["dummy"]
        #     )
            
        #     self.significances[col]["test_mean_beats_dummy"] = self.significance_test(
        #         self.scores[col]["mean"] - self.scores[col]["dummy"]
        #     )

        #     # Conservative version
        #     # TODO: Experiment with different ways to determine categorical features
        #     ### !! We only want to infer anything about the column if it is possible to predict better than the dummy classifier
        #     if self.significances[col]["test_mean_beats_dummy"]<self.alpha: # Previously, I also used LGB, but removed this behavior as it led to missing cat features when LGB simply performs poorly
        #         if self.lgb_model_type=="full-capacity":
        #             if self.significances[col]["test_lgb_superior"]<self.alpha:
        #                 self.dtypes[col] = "numeric"
        #                 comb_num.append(col)
        #             else:
        #                 self.dtypes[col] = "categorical" 
        #                 comb_bothfine.append(col)
        #         else:
        #             if self.significances[col]["test_mean_superior"]<self.alpha:
        #                 self.dtypes[col] = "categorical"
        #                 comb_cat.append(col)
        #             elif self.significances[col]["test_lgb_superior"]<self.alpha:
        #                 self.dtypes[col] = "numeric"
        #                 comb_num.append(col)
        #             else:
        #                 self.dtypes[col] = "indifferent" 
        #                 comb_bothfine.append(col)
        #     else:
        #         self.dtypes[col] = "irrelevant"
        #         irrelevant_cols.append(col)
    
        # if verbose:
        #     print("\n")
        #     print(f"According to the combination test:")
        #     print(f"...   {len(irrelevant_cols)}/{len(numeric_cand_cols)} columns are not predictive at all.")    
        #     print(f"...   {len(comb_num)}/{len(numeric_cand_cols)} columns are numeric.")    
        #     print(f"...   {len(comb_cat)}/{len(numeric_cand_cols)} columns are categorical.")    
        #     print(f"...   {len(comb_bothfine)}/{len(numeric_cand_cols)} columns are indifferent.")    

        # ### Reality check: Can transforming to categorical improve performance?
        # if self.fit_cat_models:
        #     cat_improve_cols = []
        #     for cnum, col in enumerate(comb_cat):                
        #         print(f"\rAs-categorical test: {cnum+1}/{len(comb_cat)} columns processed", end="", flush=True)
        #         x_use = X_num[[col]].copy()
            
        #         params = self.adapt_lgb_params(x_use[col], base_params.copy())

        #         model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        #         pipe = Pipeline([("model", model)])
        #         self.scores[col]["lgb-cat"] = self.cv_scores_with_early_stopping(clean_feature_names(x_use).astype("category"), y, pipe)


        #         self.significances[col]["test_cat_superior"] = self.significance_test(
        #             self.scores[col]["lgb-cat"] - self.scores[col]["lgb"]
        #         )

        #         if self.significances[col]["test_cat_superior"]<self.alpha:
        #             cat_improve_cols.append(col)
        #         else:
        #             if self.assign_numeric_with_combination:
        #                 self.dtypes[col] = "numeric"
        #             else:
        #                 self.dtypes[col] = "mean>lgb>=lgb-cat"
        
        #     if verbose:
        #         print("\n")
        #         print(f"For {len(cat_improve_cols)}/{len(comb_cat)} columns performance can be improved by transforming to categorical.")
        
        # Prepare objects to transform columns
        reassign_cols = [col for col in X.columns if self.dtypes[col]=="categorical" and self.orig_dtypes[col]!="categorical"]
        for col in reassign_cols:
            self.cat_dtype_maps[col] = pd.CategoricalDtype(categories=list(X[col].astype(str).fillna("nan").unique()))
            # TODO: CHange to use only train
        
        if self.assign_numeric:
            reassign_cols = [col for col in X.columns if self.dtypes[col]=="numeric" and self.orig_dtypes[col]!="numeric"]
            self.numeric_means = {col: X_num[col].mean() for col in reassign_cols}



    def transform(self, X_input, mode = "overwrite"):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)

        X = X_input.copy()

        for col in self.cat_dtype_maps:
            if mode == "overwrite":
                X[col] = X[col].astype(str).fillna("nan").astype(self.cat_dtype_maps[col])
            elif mode == "add":
                X[col+'_cat'] = X[col].astype(str).fillna("nan").astype(self.cat_dtype_maps[col])
                self.dtypes[col+'_cat'] = "categorical"
                self.orig_dtypes[col+'_cat'] = "categorical"
        
        if self.assign_numeric:
            reassign_cols = [col for col in X.columns if self.dtypes[col]=="numeric" and self.orig_dtypes[col]!="numeric"]
            for col in reassign_cols:
                # TODO: Implement functionality for partially coerced columns
                X[col] = pd.to_numeric(X[col], errors= 'coerce').astype(float)
                if X[col].isna().any() and self.orig_dtypes==[col]=="categorical":
                    X[col] = X[col].fillna(self.numeric_means[col])

        return X

    def get_params(self, deep=True):
        return {
            "min_q_as_num": self.min_q_as_num,
            "n_folds": self.n_folds,
            "lgb_model_type": self.lgb_model_type,
            "assign_numeric": self.assign_numeric,
            "assign_numeric_with_combination": self.assign_numeric_with_combination,
            "detect_numeric_in_string": self.detect_numeric_in_string,
            "use_highest_corr_feature": self.use_highest_corr_feature,
            "num_corr_feats_use": self.num_corr_feats_use
        }
    

if __name__ == "__main__":
    import openml
    import pandas as pd
    from sympy import rem
    from utils import get_benchmark_dataIDs, get_metadata_df
    from ft_detection import FeatureTypeDetector

    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'concrete_compressive_strength'  # (['airfoil_self_noise', 'Amazon_employee_access', 'anneal', 'Another-Dataset-on-used-Fiat-500', 'bank-marketing', 'Bank_Customer_Churn', 'blood-transfusion-service-center', 'churn', 'coil2000_insurance_policies', 'concrete_compressive_strength', 'credit-g', 'credit_card_clients_default', 'customer_satisfaction_in_airline', 'diabetes', 'Diabetes130US', 'diamonds', 'E-CommereShippingData', 'Fitness_Club', 'Food_Delivery_Time', 'GiveMeSomeCredit', 'hazelnut-spread-contaminant-detection', 'healthcare_insurance_expenses', 'heloc', 'hiva_agnostic', 'houses', 'HR_Analytics_Job_Change_of_Data_Scientists', 'in_vehicle_coupon_recommendation', 'Is-this-a-good-customer', 'kddcup09_appetency', 'Marketing_Campaign', 'maternal_health_risk', 'miami_housing', 'NATICUSdroid', 'online_shoppers_intention', 'physiochemical_protein', 'polish_companies_bankruptcy', 'APSFailure', 'Bioresponse', 'qsar-biodeg', 'QSAR-TID-11', 'QSAR_fish_toxicity', 'SDSS17', 'seismic-bumps', 'splice', 'students_dropout_and_academic_success', 'taiwanese_bankruptcy_prediction', 'website_phishing', 'wine_quality', 'MIC', 'jm1', 'superconductivity']) 

    tids, dids = get_benchmark_dataIDs(benchmark)  

    remaining_cols = {}

    for tid, did in zip(tids, dids):
        task = openml.tasks.get_task(tid)  # to check if the datasets are available
        data = openml.datasets.get_dataset(did)  # to check if the datasets are available
        if data.name!=dataset_name:
            continue
        # else:
        #     break
        print(data.name)
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



        detector = FeatureTypeDetector(target_type=target_type, 
                                    interpolation_criterion="match",  # 'win' or 'match'
                                    lgb_model_type='huge-capacity',
                                    significance_method='median',
                                    alpha=0.5,
                                    verbose=False)
        
        # detector.fit(X, y, verbose=False)
        # print(pd.Series(detector.dtypes).value_counts())
        
        rem_cols: list = detector.handle_trivial_features(X)
        print('--'*50)
        print(f"\rTrivial features removed: {X.shape[1]-len(rem_cols)}\r")
        X = X[rem_cols]
        if len(rem_cols) == 0:
            remaining_cols[data.name] = []
            continue

        X_num = X[rem_cols].astype(float).copy()
        
        rem_cols = detector.get_dummy_mean_scores(X, y)
        print('--'*50)
        print(f"\rIrrelevant features removed: {X.shape[1]-len(rem_cols)}\r")
        X = X[rem_cols]
        if len(rem_cols) == 0:
            remaining_cols[data.name] = []
            continue    

        rem_cols = detector.leave_one_out_test(X, y)
        print('--'*50)
        print(f"\rLeave-One-Out features removed: {X.shape[1]-len(rem_cols)}\r")
        X = X[rem_cols]
        if len(rem_cols) == 0:
            remaining_cols[data.name] = []
            continue

        
        rem_cols = detector.combination_test(X, y, max_binning_configs=3)
        print('--'*50)
        print(f"\rCombination features removed: {X.shape[1]-len(rem_cols)}\r")
        X = X[rem_cols]
        if len(rem_cols) == 0:
            remaining_cols[data.name] = []
            continue
        
        rem_cols = detector.interpolation_test(X, y, max_degree=3)
        print('--'*50)
        print(f"\rInterpolation features removed: {X.shape[1]-len(rem_cols)}\r")
        X = X[rem_cols]
        if len(rem_cols) == 0:
            remaining_cols[data.name] = []
            continue

        rem_cols = detector.interaction_test(X, X_num, y)
        print('--'*50)
        print(f"\rInteraction features removed: {X.shape[1]-len(rem_cols)}\r")

        # rem_cols = detector.performance_test(X, y)
        # print('--'*50)
        # print(f"\rLGB-performance features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue

        remaining_cols[data.name] = rem_cols
        print(f"{data.name} ({len(rem_cols)}): {rem_cols}")

        # if len(rem_cols) > 0:
        #     cat_res = CatResolutionDetector(target_type='regression').fit(X, y)
        #     print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))

        # if data.name=='nyc-taxi-green-dec-2016':
        #     break

        # for i, col in enumerate(X.columns):
        #     grouped_interpolation_test(X[col],(y==y[0]).astype(int), 'binary', add_dummy=True if i==0 else False)


