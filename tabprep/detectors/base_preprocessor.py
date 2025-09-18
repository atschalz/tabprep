import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline

from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor, UnivariateLinearRegressor, UnivariateLogisticClassifier, PolynomialRegressor, PolynomialLogisticClassifier, \
    LightGBMBinner, KMeansBinner, CustomLinearModel, TargetMeanClassifierCut, TargetMeanRegressorCut
from tabprep.utils.modeling_utils import adapt_lgb_params, adjust_target_format, make_cv_function, clean_feature_names

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
import time

import lightgbm as lgb
from typing import Literal

class BasePreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, target_type,
                n_folds=5, 
                alpha=0.1,
                significance_method='wilcoxon', # ['ttest', 'wilcoxon', 'median]
                mvp_criterion: Literal['significance', 'average'] = 'significance', 
                mvp_max_cols_use=100, # Maximum number of columns to use in the multivariate performance test
                random_state=42,
                verbose=True,
                lgb_model_type='default', # 'default', 'unique-based', 'huge-capacity', 'full-capacity', 'unique-based-binned', 'high-capacity', 'high-capacity-binned'
                multi_as_bin=True,  # If True, treat multi-class as binary for the purpose of interaction tests
                 ):
        # Parameters
        # NOTE: Idea: Reinvent interaction test by first categorizing a dataset into 1) there are already clear categorical features; 2) there are only numerical features; 3) there are only numerical features, but many categorical candidates. And based on that use different interaction tests.
        # TODO: Implement (optional) multi-class functionality.
        # Changes to base preprocessor parameters:
        if target_type == 'multiclass' and multi_as_bin:
            self.target_type = 'binary'
        else:
            self.target_type = target_type
        self.n_folds = n_folds
        self.alpha = alpha
        self.significance_method = significance_method
        self.mvp_criterion = mvp_criterion  # 'significance' or 'average'
        self.mvp_max_cols_use = mvp_max_cols_use  # Maximum number of columns to use in the multivariate performance test
        self.random_state = random_state
        self.verbose = verbose
        self.lgb_model_type = lgb_model_type
        self.multi_as_bin = multi_as_bin  # If True, treat multi-class as binary for the purpose of interaction tests

        # Functions
        self.cv_func = make_cv_function(target_type=self.target_type, vectorized=False, random_state=self.random_state)
        # TODO: Speedups from vectorization need to be properly evaluated before using vectorized functions
        # self.cv_scores_with_early_stopping_vec = make_cv_function(target_type=self.target_type, early_stopping_rounds=20, vectorized=True)

        if self.significance_method == 'wilcoxon':
            from tabprep.utils.eval_utils import p_value_wilcoxon_greater_than_zero
            self.significance_test = p_value_wilcoxon_greater_than_zero
        elif self.significance_method == 'ttest':
            from tabprep.utils.eval_utils import p_value_ttest_greater_than_zero
            self.significance_test = p_value_ttest_greater_than_zero
        elif self.significance_method == 'median':
            from tabprep.utils.eval_utils import p_value_sign_test_median_greater_than_zero
            self.significance_test = p_value_sign_test_median_greater_than_zero

        # Output variables
        self.scores = {}
        self.significances = {}
        self.times = {}

        self.adjust_target_format = lambda y: adjust_target_format(y,  self.target_type, multi_as_bin=self.multi_as_bin)

    def get_dummy_mean_scores(self, X, y, return_preds=False):
        X_use = X.copy()

        if return_preds:
            preds = X_use.copy()

        for cnum, col in enumerate(X_use.columns):            
            x_use = X_use[col].copy()
            self.scores[col] = {}
            self.significances[col] = {}
            if self.verbose:
                print(f"\rDummy and target mean test: {cnum+1}/{len(X_use.columns)} columns processed", end="", flush=True)
            self.significances[col] = {}

            # dummy baseline on single column
            dummy_pipe = Pipeline([("model", DummyRegressor(strategy='mean') if self.target_type=="regression" else DummyClassifier(strategy='prior'))])
            self.scores[col]["dummy"] = self.cv_func(x_use.to_frame(), y, dummy_pipe)['scores']
            
            model = (TargetMeanClassifier() if self.target_type=="binary"
                    else TargetMeanRegressor())
            pipe = Pipeline([("model", model)])
            if return_preds:
                self.scores[col]["mean"], preds_ = self.cv_func(x_use.to_frame(), y, pipe, return_preds=True)['scores']
                preds[col] = preds_
            else:
                self.scores[col]["mean"] = self.cv_func(x_use.to_frame(), y, pipe)['scores']

            self.significances[col]["test_irrelevant_mean"] = self.significance_test(
                self.scores[col]["mean"] - self.scores[col]["dummy"]
            )

    # TODO: Add derange test using the function in utils.misc
    # TODO: Add grouped interpolation test from utils.misc

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

        return self.cv_func(x.to_frame(), y, pipe)['scores']
    
    def single_combination_test(
            self,
            x, y, 
            max_bin=255,
            binning_strategy='lgb',  # 'lgb', 'KMeans', 'DT'
    ):        
        if not x.dtype in ["int", "float", "bool"]:
            x = pd.to_numeric(x, errors='coerce')

        if binning_strategy == 'lgb':
            pipe = Pipeline([
                ("binning", LightGBMBinner(max_bin=max_bin)),
                ("model", TargetMeanClassifier() if self.target_type == "binary" else TargetMeanRegressor())
            ])
        elif binning_strategy == 'KMeans':
            pipe = Pipeline([
                ("binning", KMeansBinner(max_bin=max_bin)),
                ("model", TargetMeanClassifier() if self.target_type == "binary" else TargetMeanRegressor())
            ])
        elif binning_strategy == 'DT':
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
            raise ValueError(f"Unsupported binning strategy: {binning_strategy}. Use 'lgb', 'KMeans' or 'DT'.")            

        return self.cv_func(x.to_frame(), y, pipe)['scores']
    
    def cat_threshold_test(self, X, y, early_stopping=True, verbose=False):
        X_use = X.copy()
        remaining_cols = X_use.columns.tolist()
        # for q in np.linspace(0.1,1,max_binning_configs, endpoint=False):

        if self.target_type == 'regression':
            target_cut_model = lambda t: TargetMeanRegressorCut(t)
        else:
            target_cut_model = lambda t: TargetMeanClassifierCut(t)

        for cnum, col in enumerate(X_use.columns):            
            if cnum==0:
                minus_done = 0
            possible_thresholds = X_use[col].value_counts(ascending=True).unique()
            for thresh in possible_thresholds:
                if early_stopping and len(remaining_cols)==0:
                    break                
                if early_stopping and col not in remaining_cols: # early_stopping=True is used if we just want to quickly find numeric features. If best performance matters, we would rather try all bins
                    minus_done += 1
                    continue                
                if verbose:
                    print(f"\rCat-Threshold test with max_bin of {thresh}: {cnum-minus_done+1}/{len(remaining_cols)} columns processed", end="", flush=True)

                x_use = X_use[col].copy()
                pipe = Pipeline([
                    ("model", target_cut_model(thresh))
                ])
                self.scores[col][f"cat_threshold_test_{thresh}"] = self.cv_func(X[[col]], y, pipe)['scores']

                self.significances[col][f"test_cat_threshold_test_{thresh}_superior"] = self.significance_test(
                        self.scores[col][f"cat_threshold_test_{thresh}"] - self.scores[col]["mean"]
                    )

                self.significances[col][f"test_mean_superior_to_cat_threshold{thresh}"] = self.significance_test(
                        self.scores[col]["mean"] - self.scores[col][f"cat_threshold_test_{thresh}"]
                    )

                self.significances[col][f"cat_threshold{thresh}_superior_to_dummy"] = self.significance_test(
                        self.scores[col][f"cat_threshold_test_{thresh}"] - self.scores[col]["dummy"]
                    )
                
                if early_stopping and self.significances[col][f"test_mean_superior_to_cat_threshold{thresh}"] < self.alpha:
                    remaining_cols.remove(col)
                    break

    def get_target_rep_type(self, X, y):
        
        # 1. Get all dummy-mean scores
        self.get_dummy_mean_scores(X, y)

        X_cat = X.select_dtypes(include=['object', 'category'])
        X_num = X.select_dtypes(exclude=['object', 'category'])

        assert X_cat.shape[1] + X_num.shape[1] ==  X.shape[1], "Not all features covered through num/cat selection."

        if X_num.shape[1] > 0:
            # 2. Get all combination scores
            comb_cols = self.combination_test(X_num, y, early_stopping=False)
            
            # 3. Get linear scores
            linear_cols = self.interpolation_test(X_num, y, max_degree=1)

        if X_cat.shape[1] > 0:
            self.cat_threshold_test(X_cat, y)

        for col in X.columns:
            self.feature_target_rep[col] = pd.Series(pd.DataFrame(self.scores[col]).mean().sort_values()).idxmax()
            
        num_as_cat = [self.feature_target_rep[col]=='mean' for col in X.columns]
        if any(num_as_cat):
            self.cat_threshold_test(X.loc[: , num_as_cat], y)

    def cat_as_num(self, X_in):
        X_out = X_in.copy()
        for col in X_out.select_dtypes(include=['object', 'category']).columns:
            num_convertible = pd.to_numeric(X_out[col].dropna(), errors='coerce').notna().all()
            if num_convertible:
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce')
            else:
                X_out[col] = OrdinalEncoder().fit_transform(X_out[[col]]).flatten()
        return X_out

    def adapt_lgb_params(self, X: pd.DataFrame, y: pd.Series = None, base_params: dict=None, lgb_model_type=None, target_type=None):
        if lgb_model_type is None:
            lgb_model_type = self.lgb_model_type
        if target_type is None:
            target_type = self.target_type
        
        if base_params is None:

            params = {
                "objective": target_type,
                "boosting_type": "gbdt",
                "n_estimators": 1000,
                'min_samples_leaf': 2,
                "max_depth": 5,
                "verbosity": -1
            }

            if target_type == "multiclass":
                params["num_class"] = len(y.unique())


        else:
            params = base_params.copy()
        
        if lgb_model_type=="default":
            pass
        elif lgb_model_type=="irrelevant":
            if X.shape[1]>=10 and X.shape[1]<100:
                params["feature_fraction"] = 0.8
            elif X.shape[1]>=100 and X.shape[1]<1000:
                params["feature_fraction"] = 0.6
            elif X.shape[1]>=1000:
                params["feature_fraction"] = 0.4
            else:
                params["feature_fraction"] = 1.0
        elif lgb_model_type=="numint":
            # params["max_bin"] = X.nunique().max() #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            # params["max_depth"] = 2
            # params["n_estimators"] = 1000
            # params["min_samples_leaf"] = 5
            params = {
                "objective": "binary" if self.target_type=="binary" else "regression",
                "boosting_type": "gbdt",
                "n_estimators": 1000,
                "verbosity": -1
            }
        elif lgb_model_type=="quantile_1":
            params["objective"] = "quantile"
            params["alpha"] = 0.1  # median
        elif lgb_model_type=="quantile_3":
            params["objective"] = "quantile"
            params["alpha"] = 0.3  # median
        elif lgb_model_type=="quantile_5":
            params["objective"] = "quantile"
            params["alpha"] = 0.5  # median
        elif lgb_model_type=="quantile_9":
            params["objective"] = "quantile"
            params["alpha"] = 0.9  # median
        elif lgb_model_type=="fast":
            params["max_depth"] = 2
            params["n_estimators"] = 100
        elif lgb_model_type=="catint":
            params["max_depth"] = 2
        elif lgb_model_type=="fine_granular":
            params["max_bin"] = 128#000000 #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["min_samples_leaf"] = 30 # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif lgb_model_type=="unique-based":
            params["max_bin"] = X.nunique().max() #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 2 #if mode=="cat" else 2
            params["n_estimators"] = X.nunique().max() # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif lgb_model_type=="huge-capacity":
            params["max_bin"] = int(X.nunique().max()) #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["min_data_in_leaf"] = 1 #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 5 #if mode=="cat" else 2
            params["n_estimators"] = 10000 # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif lgb_model_type=="full-capacity":
            params["max_bin"] = int(X.nunique().max()) #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["min_data_in_leaf"] = 1 #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 2 #if mode=="cat" else 2
            params["n_estimators"] = int(X.nunique().max()*2) # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif lgb_model_type=="unique-based-binned":
            params["max_bin"] = int(X.nunique().max()/2) #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 2 #if mode=="cat" else 2
            params["n_estimators"] = X.nunique().max() # 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif lgb_model_type=="high-capacity":
            params["max_bin"] = X.nunique().max() #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)
        elif lgb_model_type=="high-capacity-binned":
            params["max_bin"] = int(X.nunique().max()/2) #min([10000, X_num[col].nunique()]) #if mode=="cat" else 2
            params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)
        else:
            raise ValueError(f"Unknown lgb_model_type: {self.lgb_model_type}. Use 'unique-based', 'high-capacity', 'high-capacity-binned', 'full-capacity' or 'default'.")
        
        return params

    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
                # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
            if mode == 'backward': 
                X_out = X_out.drop(test_cols, axis=1)
            elif mode == 'forward':
                pass
        else:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                X_out = X_out.drop(col, axis=1)
            elif mode == 'forward':
                X_out = X_out.drop(set(test_cols) - {col}, axis=1)
            
        return X_out

    # TODO: Probably, we can safely remove this function for now
    def get_lgb_performance(self, X_cand_in, y_in, lgb_model_type=None, 
                            custom_prep=None, residuals=None, scale_y=None,
                            reg_assign_closest_y=False,
                            ):
        X = X_cand_in.copy()
        y = y_in.copy()
        
        # TODO: Write a 'preprocess_for_lgb' function using AG
        # Necessary preprocessing
        obj_cols = X.select_dtypes(include=['object']).columns.tolist()
        X[obj_cols] = X[obj_cols].astype('category')
        # Adapt data adnd train models
        if residuals is not None:
            params = adapt_lgb_params(target_type='regression', lgb_model_type=lgb_model_type, X=X, y=y)
            model = lgb.LGBMRegressor(**params)
            original_y = y.copy()
            if isinstance(residuals,str) and residuals == 'linear_residuals':
                y_use = y.copy()
            elif isinstance(residuals,pd.Series):
                y_use = residuals
            else:
                raise ValueError(f"Unknown residuals option: {residuals}. Use 'linear_residuals' or provide a pd.Series with the residuals.")
        else:
            params = adapt_lgb_params(target_type=self.target_type, lgb_model_type=lgb_model_type, X=X, y=y)
            model = lgb.LGBMClassifier(**params) if self.target_type in ["binary", 'multiclass'] else lgb.LGBMRegressor(**params)
            original_y = None
            y_use = y
        pipe = Pipeline([("model", model)])

        return self.cv_func(clean_feature_names(X), y_use, pipe, 
                            return_importances=True, return_preds=True, custom_prep=custom_prep, 
                            original_y=original_y, scale_y=scale_y,
                            reg_assign_closest_y=reg_assign_closest_y
                            
                            )

    def multivariate_performance_test(self, X_cand_in, y_in, 
                                      test_cols, suffix='mvp',  
                                      max_cols_use=100,
                                      individual_test_condition='always' # 'always', 'significant', 'never'


                                      ):
        '''
        X_cand_in: pd.DataFrame - The candidate data with raw input and possibly additional candidate columns.
        y_in: pd.Series - The target variable.
        test_cols: list - The columns to test in X_cand.
        suffix: str - The suffix to use for naming the results.
        max_cols_use: int - The maximum number of columns from X_raw to use
        '''
        X_cand = X_cand_in.copy()
        y = y_in.copy()
        np.random.seed(42)
        
        if X_cand.shape[1]-len(test_cols) > max_cols_use:
            drop_cols = np.random.choice(X_cand.drop(test_cols, axis=1, errors='ignore').columns, size=X_cand.shape[1]-max_cols_use-len(test_cols), replace=False)
            X_cand = X_cand.drop(drop_cols, axis=1, errors='ignore')
            prefix = f"LGB-subsample{X_cand.shape[1]}"
        else:
            prefix = "LGB-full"
        
        # TODO: Change to get_lgb_params
        params = adapt_lgb_params(self.target_type, X_cand[X_cand.nunique().idxmax()])

        self.scores[prefix] = {}
        self.significances[prefix] = {}
        if self.verbose:
            print(f"\rMultivariate performance test with {len(test_cols)} columns.")
        # Get performance for all base columns
        X_use = X_cand.copy()
        # TODO: Write a 'preprocess_for_lgb' function using AG
        # Necessary preprocessing
        obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
        X_use[obj_cols] = X_use[obj_cols].astype('category')
        # Adapt data adnd train models
        X_use = self.adapt_for_mvp_test(X_use, col=None, test_cols=test_cols, mode='backward')
        model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        pipe = Pipeline([("model", model)])
        self.scores[prefix]['raw'] = self.cv_func(clean_feature_names(X_use), y, pipe, return_iterations=False)['scores']
        
        ### Get performance for all columns with interactions
        X_use = X_cand.copy() 
        # X_use = X_use.astype('float')  # Use category dtype for categorical columns
        # Necessary preprocessing
        obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
        X_use[obj_cols] = X_use[obj_cols].astype('category')
        # Adapt data and train models
        X_use = self.adapt_for_mvp_test(X_use, col=None, test_cols=test_cols, mode='forward')
        model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
        pipe = Pipeline([("model", model)])
        self.scores[prefix][f"{len(test_cols)}-{suffix}"] = self.cv_func(clean_feature_names(X_use), y, pipe)['scores']

        self.significances[prefix][f"{len(test_cols)}-{suffix}"] = self.significance_test(self.scores[prefix][f"{len(test_cols)}-{suffix}"]-self.scores[prefix]['raw'])

        if self.mvp_criterion=='significance':
            significant = self.significances[prefix][f"{len(test_cols)}-{suffix}"] < self.alpha
        elif self.mvp_criterion=='average':
            significant = np.mean(self.scores[prefix][f"{len(test_cols)}-{suffix}"] - self.scores[prefix]['raw']) > 0
        else:
            raise ValueError(f"Unknown mvp_criterion: {self.mvp_criterion}. Use 'significance' or 'average'.")
        
        if individual_test_condition == 'always':
            perform_individual_tests = True
        elif individual_test_condition == 'significant':
            if self.significances[prefix][f"{len(test_cols)}-{suffix}"] < 1:
                perform_individual_tests = True
            else:
                perform_individual_tests = False
        elif individual_test_condition == 'never':
            perform_individual_tests = False

        if significant:
            if self.verbose:
                print(f"ALL-{suffix} is significantly better than using the raw columns with p-value {self.significances[prefix][f'{len(test_cols)}-{suffix}']:.3f}")
                print(f"Execute per-column tests in backward selection mode.")
            # for col in test_cols:
            #     self.dtypes[col] = 'categorical'
            ref_config = f"{len(test_cols)}-{suffix}"
            selection_mode = 'backward'
            suffix = f'NO{suffix}'
        else:
            if self.verbose:
                print(f"ALL-{suffix} is not significantly better than using the raw columns with p-value {self.significances[prefix][f'{len(test_cols)}-{suffix}']:.3f}")
                print(f"Execute per-column tests in forward selection mode.")
            # for col in test_cols:
            #     self.dtypes[col] = 'numeric'
            ref_config = "raw"
            selection_mode = 'forward'
                
        accepted_cols = []
        if perform_individual_tests and len(test_cols) > 1:
            use_scores = self.scores[prefix][ref_config]
            for num, col in enumerate(test_cols):
                # TODO: Add early stopping and sort the columns to test in a reasonable way
                if self.verbose:
                    print(f"\rColumn {num+1}/{len(test_cols)}: {col}", end="", flush=True)
                X_use = X_cand.copy()
                obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
                X_use[obj_cols] = X_use[obj_cols].astype('category')
                X_use = self.adapt_for_mvp_test(X_use, col=col, test_cols=test_cols, mode=selection_mode)
                # if selection_mode == 'backward':
                #     use_scores = self.scores[prefix][f"{len(test_cols)}-{suffix}"]
                #     X_use = self.adapt_col_for_mvp_test(X_use, col=col, test_cols=test_cols, mode=selection_mode)
                #     # X_use[test_cols] = X_use[test_cols].astype('category')  # Use category dtype for categorical columns
                #     # X_use[col] = X_use[col].astype(float)  
                #     # suffix = "-asnum"
                # else:
                #     use_scores = self.scores[prefix]['raw']
                #     X_use[test_cols] = X_use[test_cols].astype(float)  # Use category dtype for categorical columns
                #     X_use[col] = X_use[col].astype('category')
                #     suffix = "-ascat"
                # TODO: Make sure to prevent leaks with category dtypes
                model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
                pipe = Pipeline([("model", model)])
                self.scores[prefix][f"{col}{suffix}"] = self.cv_func(clean_feature_names(X_use), y, pipe)['scores']

                self.significances[prefix][f"{col}{suffix}_superior_{ref_config}"] = self.significance_test(self.scores[prefix][f"{col}{suffix}"]-use_scores)

                if self.mvp_criterion=='significance':
                    significant = self.significances[prefix][f"{col}{suffix}_superior_{ref_config}"] < self.alpha
                elif self.mvp_criterion=='average':
                    significant = np.mean(self.scores[prefix][f"{col}{suffix}"]-use_scores) > 0
                else:
                    raise ValueError(f"Unknown mvp_criterion: {self.mvp_criterion}. Use 'significance' or 'average'.")

                if not significant and selection_mode == 'backward':
                    # TODO: Add more flexible logic for backward elimination when dropping the candidate does not decrease performance
                    accepted_cols.append(col)
                elif significant and selection_mode == 'forward':
                    accepted_cols.append(col)

                if significant and self.verbose:
                    print(f"{col}{suffix} is significantly better in {selection_mode} mode with p-value {self.significances[prefix][f'{col}{suffix}_superior_{ref_config}']:.3f}")
                    # self.dtypes[col] = 'numeric' if selection_mode == 'backward' else 'categorical'
                # if self.dtypes[col] == 'categorical':
                #     accepted_cols.append(col)
            if self.verbose:
                print(f"\r{len(accepted_cols)}/{len(test_cols)} candidate columns are accepted.")
        elif selection_mode == 'backward': # backward indicates that the candidate significantly improved
            accepted_cols.extend(test_cols)
        
        return accepted_cols

    def run_test(self, X, y, test_name, test_cols=list(), **kwargs):
        start = time.time()
        if test_name == 'dummy_mean':
            remaining_cols = self.get_dummy_mean_scores(X, y)
        elif test_name == 'multivariate_performance':
            remaining_cols = self.multivariate_performance_test(X, y, test_cols, max_cols_use=self.mvp_max_cols_use)
        else:
            print(f"Unknown test_name: {test_name}. Use 'dummy_mean' or implement a new test.")
            remaining_cols = X.columns.values.tolist()
        self.times[test_name] = time.time() - start
        
        if self.verbose:
            print(f"\r{len(remaining_cols)/len(X.columns):.2%} of the candidate columns remain after the {test_name} test.")
        
        return remaining_cols

    def fit(self, X_input, y_input):
        return self

    def transform(self, X_input):
        
        return X_input

    def get_params(self, deep=True):
        return {
            "target_type": self.target_type,
            "n_folds": self.n_folds,
            "alpha": self.alpha,
            "significance_method": self.significance_method,
            "verbose": self.verbose
        }