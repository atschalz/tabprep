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
    def __init__(self, target_type, 
                tests_to_run=[
                'dummy_mean',
                'leave_one_out',
                'combination', 
                'interpolation', 
                # 'mode', # Likely is the same as low max_bins in combination test, therefore redundant
                # 'interaction', 
                # 'performance', 
                            ], # Trivial features test always runs

                # Global hyperparameters for each test
                min_q_as_num=6, 
                n_folds=5, 
                alpha=0.05,
                significance_method='wilcoxon', # ['t-test', 'wilcoxon', 'median]
                verbose=True,
                assign_numeric=False, 

                # Trivial feature test hyperparameters
                detect_numeric_in_string=False, 

                # Interpolation test hyperparameters
                max_degree=3, # max degree of polynomial interpolation
                interpolation_criterion="match", # ['win', 'match']

                # Combination test hyperparameters
                combination_criterion='win',
                combination_test_min_bins=16,
                combination_test_max_bins=2048,
                assign_numeric_with_combination=False,
                binning_strategy='lgb', # ['lgb', 'KMeans', 'DT']

                # Mode test hyperparameters
                max_modes=5,

                # Interaction test hyperparameters
                interaction_mode='high_corr', # ['high_corr', 'all' int]

                 # Performance test hyperparameters
                 lgb_model_type="unique-based-binned", 
                 fit_cat_models=True,

                 use_performance_test=False,
                 use_interaction_test=True,
                 use_leave_one_out_test=False,
                 
                 # experimental
                 use_highest_corr_feature=False, num_corr_feats_use=0,
                 drop_modes=0,
                 drop_unique=False,
                 ):
        # TODO: Adjust print statements to look nicer
        # Parameters
        self.target_type = 'binary' if target_type == 'multiclass' else target_type
        self.tests_to_run = tests_to_run
        self.min_q_as_num = min_q_as_num
        self.n_folds = n_folds
        self.lgb_model_type = lgb_model_type
        self.max_degree = max_degree
        self.max_modes = max_modes
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
        self.binning_strategy = binning_strategy
        self.combination_criterion = combination_criterion
        self.drop_unique = drop_unique
        self.alpha = alpha
        self.interaction_mode = interaction_mode
        self.use_interaction_test = use_interaction_test
        self.use_leave_one_out_test = use_leave_one_out_test
        self.drop_modes = drop_modes
        self.significance_method = significance_method
        self.verbose = verbose

        # Currently dummy-mean test is necessary as all other tests build on it
        if 'dummy_mean' not in self.tests_to_run:
            self.tests_to_run.insert(0, 'dummy_mean')

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
        self.removed_after_test = {}
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
        elif interpolation_method in [f'poly{d}' for d in range(2, 100)]:
            degree = int(interpolation_method[4:])
            interpol_model = PolynomialLogisticClassifier(degree=degree) if self.target_type == "binary" else PolynomialRegressor(degree=degree)
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
            elif d in range(2,100) and type(d) is int:
                interpolation_method = f'poly{d}'
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

        if self.binning_strategy == 'lgb':
            pipe = Pipeline([
                ("binning", LightGBMBinner(max_bin=max_bin)),
                ("model", TargetMeanClassifier() if self.target_type == "binary" else TargetMeanRegressor())
            ])
        elif self.binning_strategy == 'KMeans':
            from utils import KMeansBinner
            pipe = Pipeline([
                ("binning", KMeansBinner(max_bin=max_bin)),
                ("model", TargetMeanClassifier() if self.target_type == "binary" else TargetMeanRegressor())
            ])
        elif self.binning_strategy == 'DT':
            if self.target_type == "regression":
                from sklearn.tree import DecisionTreeRegressor as tree_binner
            else:
                from sklearn.tree import DecisionTreeClassifier as tree_binner
            pipe = Pipeline([
                ("model", tree_binner(
                    max_leaf_nodes = max_bin,     # desired number of bins
                    min_samples_leaf = 1,   # min pts per bin to avoid overfitting
                    # criterion = "entropy"   # or "gini"
                ))
            ])
        else:
            raise ValueError(f"Unsupported binning strategy: {self.binning_strategy}. Use 'lgb', 'KMeans' or 'DT'.")            

        return self.cv_scores_with_early_stopping(x.to_frame(), y, pipe)
    
    def combination_test(self, X, y, max_binning_configs=3, early_stopping=True, assign_dtypes = True, verbose=False):
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
                if verbose:
                    print(f"\rCombination test with max_bin of {m_bin}: {cnum-minus_done+1}/{len(remaining_cols)} columns processed", end="", flush=True)

                x_use = X_use[col].copy()
                self.scores[col][f"combination_test_{m_bin}"] = self.single_combination_test(x_use, y, max_bin=m_bin)
                
                self.significances[col][f"test_combination_test_{m_bin}_superior"] = self.significance_test(
                        self.scores[col][f"combination_test_{m_bin}"] - self.scores[col]["mean"]
                    )
                
                self.significances[col][f"test_mean_superior_to_combination{m_bin}"] = self.significance_test(
                        self.scores[col]["mean"] - self.scores[col][f"combination_test_{m_bin}"]
                    )


                if self.combination_criterion == "win":
                    if self.significances[col][f"test_combination_test_{m_bin}_superior"]<self.alpha:
                        if assign_dtypes:
                            self.dtypes[col] = "numeric"
                        comb_cols.append(col)
                elif self.combination_criterion == "match" and m_bin==m_bins[-1]:
                    if self.significances[col][f"test_mean_superior_to_combination{m_bin}"]>self.alpha:
                        if assign_dtypes:
                            self.dtypes[col] = "numeric"
                        comb_cols.append(col)


            if verbose:
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
        cat_improve_cols = []
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
                continue
            
            if self.fit_cat_models:
                model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
                pipe = Pipeline([("model", model)])
                self.scores[col]["lgb-cat"] = self.cv_scores_with_early_stopping(clean_feature_names(x_use).astype("category"), y, pipe)

                self.significances[col]["test_lgb-ascat_superior"] = self.significance_test(
                    self.scores[col]["lgb-cat"] - self.scores[col]["lgb"]
                )
                
                self.significances[col]["test_lgb-asnum_superior"] = self.significance_test(
                     self.scores[col]["lgb"] - self.scores[col]["lgb-cat"]
                )

                if self.significances[col]["test_lgb-ascat_superior"]<self.alpha:
                    cat_improve_cols.append(col)
                    self.dtypes[col] = "categorical"
                elif self.significances[col]["test_lgb-asnum_superior"]<self.alpha:
                    perf_cols.append(col)
                    self.dtypes[col] = "numeric"

        if self.verbose:
            print("\n")
            print(f"{len(perf_cols)}/{len(X.columns)} columns are numeric acc. to performance test.")
            if self.fit_cat_models:
                print(f"{len(cat_improve_cols)}/{len(X.columns)} columns are categorical acc. to performance test.")
        remaining_cols = [x for x in X.columns if x not in perf_cols]
        
        if self.fit_cat_models:
            remaining_cols = [x for x in remaining_cols if x not in cat_improve_cols]
        
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
    
    def interaction_test(self, X_cand, y, X_full):
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
        base_cors = X_full.corrwith(y).abs()

        # Reduce to 100 possible features to speed things up
        if base_cors.shape[0]>100:
            base_cors = base_cors.sort_values(ascending=False).head(100)
            X_full = X_full.loc[:, base_cors.index]

        interaction_cols = []
        for cnum, col in enumerate(X_cand.columns):
            if self.verbose:
                print(f"\rInteraction test for column {cnum+1}/{len(X_cand.columns)} columns processed", end="", flush=True)

            ### 1. Get Interactions
            cols_use = [c for c in X_full.columns if c != col]
            
            X_int = pd.DataFrame(
                index=X_full.index, 
                columns=[f"{col}_/_{col2}" for col2 in cols_use]+ \
                [f"{col}_x_{col2}" for col2 in cols_use]+ \
                [f"{col}_-_{col2}" for col2 in cols_use]+ \
                [f"{col}_+_{col2}" for col2 in cols_use]
            )
            
            # TODO: Think about whether reversing - and / makes sense
            # TODO: Think about whether setting nans for division by zero is appropriate
            X_int[[f"{col}_/_{col2}" for col2 in cols_use]] = (X_cand[[col]*len(cols_use)] / X_full[cols_use].replace(0, np.nan).values).values
            X_int[[f"{col}_x_{col2}" for col2 in cols_use]] = (X_cand[[col]*len(cols_use)] * X_full[cols_use].values).values
            X_int[[f"{col}_-_{col2}" for col2 in cols_use]] = (X_cand[[col]*len(cols_use)] - X_full[cols_use].values).values
            X_int[[f"{col}_+_{col2}" for col2 in cols_use]] = (X_cand[[col]*len(cols_use)] + X_full[cols_use].values).values

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
                self.combination_test(X_use, y, early_stopping=False, assign_dtypes=False, verbose=False)
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
    
    def mode_test(self, X_in, y):
        X = X_in.copy()

        if self.target_type == "regression":
            target_model = TargetMeanRegressor()
        else:
            target_model = TargetMeanClassifier()

        mode_cols = []
        for cnum, col in enumerate(X.columns):                
            print(f"\Mode test: {cnum+1}/{len(X.columns)} columns processed", end="", flush=True)
            x = X[col]
            modes = x.value_counts(dropna=False)
            for num_modes in range(1,self.max_modes+1):
                mode_map = dict(zip(modes.index[:num_modes], range(num_modes)))
                x_use = x.map(mode_map).fillna(num_modes)
                # roc_auc_score(y, TargetMeanClassifier().fit(X[[x.name]]==0, y).predict_proba(X[[x.name]]==0)[:,1])
                self.scores[col][f'mode{num_modes}'] = self.cv_scores_with_early_stopping(x_use.to_frame(), y, Pipeline([('model',  target_model)]))

                self.significances[col][f"{num_modes}_superior_mean"] = self.significance_test(
                    self.scores[col][f'mode{num_modes}'] - self.scores[col]['mean']
                )

                if self.significances[col][f"{num_modes}_superior_mean"]<0.05:
                    self.dtypes[col] = "multimodal"
                    mode_cols.append(col)
                    break

        remaining_cols = [x for x in X.columns if x not in mode_cols]

        return remaining_cols         

    def run_test(self, X, y, test_name, **kwargs):
        if test_name == 'dummy_mean':
            remaining_cols = self.get_dummy_mean_scores(X, y)
        elif test_name == 'leave_one_out':
            remaining_cols = self.leave_one_out_test(X, y)
        elif test_name == 'combination':
            remaining_cols = self.combination_test(X, y, verbose=self.verbose)
        elif test_name == 'interpolation':
            remaining_cols = self.interpolation_test(X, y, max_degree=self.max_degree)
        elif test_name == 'interpolation-log':
            remaining_cols = self.interpolation_test(np.log(X+1e-5), y, max_degree=self.max_degree)
        elif test_name == 'interaction':
            if 'X_full' in kwargs:
                X_full = kwargs['X_full']
            else:
                X_full = X_num.copy()
            remaining_cols = self.interaction_test(X, y, X_full=X_full)
        elif test_name == 'performance':
            remaining_cols = self.performance_test(X, y)
        elif test_name == 'mode':
            remaining_cols = self.mode_test(X, y)
        else:
            print(f"Unknown test_name: {test_name}. Use 'dummy_mean', 'leave_one_out', 'combination', 'interpolation', 'interaction' or 'performance'.")
            remaining_cols = X.columns.values.tolist()
        
        for col in set(X.columns) - set(remaining_cols):
            self.removed_after_tests[col] = test_name
        if self.verbose:
            print(f"\r{len(remaining_cols)/len(X.columns):.2%} of the candidate columns remain after the {test_name} test.")
        return remaining_cols


    def fit(self, X_input, y_input=None, verbose=False):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)
        if not isinstance(y_input, pd.DataFrame):
            y_input = pd.Series(y_input)
        X = X_input.copy()
        y = y_input.copy()
        
        self.orig_dtypes = {col: "categorical" if dtype in ["object", "category", str] else "numeric" for col,dtype in dict(X.dtypes).items()}
        for col in self.orig_dtypes:
            if X[col].nunique()<=2:
                self.orig_dtypes[col] = "binary"

        if self.target_type=="multiclass":
            # TODO: Fix this hack
            y = (y==y.value_counts().index[0]).astype(int)  # make it binary
            self.target_type = "binary"
        elif self.target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
            y = (y==y.value_counts().index[0]).astype(int)  # make it numeric

        self.col_names = X.columns
        self.dtypes = {col: "None" for col in  self.col_names}
        self.removed_after_tests = {col: "None" for col in self.col_names}

        numeric_cand_cols: list[str] = self.handle_trivial_features(X, verbose=verbose) # type: ignore
        for col in set(X.columns) - set(numeric_cand_cols):
            self.removed_after_tests[col] = "trivial"
        if len(numeric_cand_cols)==0:
            return

        if self.verbose:
            print(f"{len(numeric_cand_cols)} columns left for numeric/categorical detection")
        X_num = X[numeric_cand_cols].copy().astype(float)
        X_full = X_num.copy()


        ### TEST NUMERIC CANDIDATE COLUMNS
        for test_name in self.tests_to_run:
            if self.verbose:
                print(f"Running test: {test_name}")
            numeric_cand_cols = self.run_test(X_num, y, test_name, X_full=X_full)
            if len(numeric_cand_cols)==0:
                return
            else:
                X_num = X_num[numeric_cand_cols].copy()

        # Simply assign all remaining numeric candidate columns as categorical
        # TODO: Experiment with different ways to determine categorical features
        for col in numeric_cand_cols:
            self.dtypes[col] = "categorical"
        
        # Prepare objects to transform columns
        reassign_cols = [col for col in X.columns if self.dtypes[col]=="categorical" and self.orig_dtypes[col]!="categorical"]
        for col in reassign_cols:
            self.cat_dtype_maps[col] = pd.CategoricalDtype(categories=list(X[col].astype(str).fillna("nan").unique()))
            # TODO: Might need to change to use only train / make the whole approach use accept a train/val/test split

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
            "target_type": self.target_type,
            "tests_to_run": self.tests_to_run,
            "max_degree": self.max_degree,
            "interpolation_criterion": self.interpolation_criterion,
            "combination_criterion": self.combination_criterion,
            "combination_test_min_bins": self.combination_test_min_bins,
            "combination_test_max_bins": self.combination_test_max_bins,
            "binning_strategy": self.binning_strategy,
            "lgb_model_type": self.lgb_model_type,
            "significance_method": self.significance_method,
            "alpha": self.alpha,
            "verbose": self.verbose,
            "assign_numeric": self.assign_numeric,
            "fit_cat_models": self.fit_cat_models,

        }

    

if __name__ == "__main__":
    import openml
    import pandas as pd
    from sympy import rem
    from utils import get_benchmark_dataIDs, get_metadata_df
    from ft_detection import FeatureTypeDetector

    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'Fail'  # (['airfoil_self_noise', 'Amazon_employee_access', 'anneal', 'Another-Dataset-on-used-Fiat-500', 'bank-marketing', 'Bank_Customer_Churn', 'blood-transfusion-service-center', 'churn', 'coil2000_insurance_policies', 'concrete_compressive_strength', 'credit-g', 'credit_card_clients_default', 'customer_satisfaction_in_airline', 'diabetes', 'Diabetes130US', 'diamonds', 'E-CommereShippingData', 'Fitness_Club', 'Food_Delivery_Time', 'GiveMeSomeCredit', 'hazelnut-spread-contaminant-detection', 'healthcare_insurance_expenses', 'heloc', 'hiva_agnostic', 'houses', 'HR_Analytics_Job_Change_of_Data_Scientists', 'in_vehicle_coupon_recommendation', 'Is-this-a-good-customer', 'kddcup09_appetency', 'Marketing_Campaign', 'maternal_health_risk', 'miami_housing', 'NATICUSdroid', 'online_shoppers_intention', 'physiochemical_protein', 'polish_companies_bankruptcy', 'APSFailure', 'Bioresponse', 'qsar-biodeg', 'QSAR-TID-11', 'QSAR_fish_toxicity', 'SDSS17', 'seismic-bumps', 'splice', 'students_dropout_and_academic_success', 'taiwanese_bankruptcy_prediction', 'website_phishing', 'wine_quality', 'MIC', 'jm1', 'superconductivity']) 

    tids, dids = get_benchmark_dataIDs(benchmark)  

    remaining_cols = {}

    for tid, did in zip(tids, dids):
        task = openml.tasks.get_task(tid)  # to check if the datasets are available
        data = openml.datasets.get_dataset(did)  # to check if the datasets are available
        if dataset_name not in data.name:
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
                                    tests_to_run=['dummy_mean', 
                                                  'leave_one_out', 
                                                  'combination', 
                                                  'interpolation', 
                                                #   'interpolation-log',
                                                #   'mode',
                                                  'interaction', 
                                                #   'performance'
                                                  ],
                                    max_degree=5,
                                    interaction_mode='high_corr', 
                                    interpolation_criterion="match",  # 'win' or 'match'
                                    combination_criterion='win',
                                    combination_test_min_bins=2,
                                    combination_test_max_bins=2048,
                                    binning_strategy='DT',  # 'lgb', 'KMeans' or 'DT'
                                    # lgb_model_type='huge-capacity',
                                    significance_method='wilcoxon',
                                    alpha=0.05,
                                    verbose=True
                                    
                                    )
        
        detector.fit(X, y, verbose=False)
        rem_cols = list(detector.cat_dtype_maps.keys())
        # print(pd.Series(detector.dtypes).value_counts())

        remaining_cols[data.name] = rem_cols
        print(pd.Series(detector.removed_after_tests).value_counts())
        print(f"{data.name} ({len(rem_cols)}): {rem_cols}")
        continue
