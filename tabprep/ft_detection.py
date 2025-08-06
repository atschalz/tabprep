import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, log_loss
from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor, UnivariateLinearRegressor, UnivariateLogisticClassifier, PolynomialLogisticClassifier, PolynomialRegressor, UnivariateThresholdClassifier, MultiFeatureTargetMeanClassifier, MultiFeatureUnivariateLogisticClassifier, LightGBMBinner, KMeansBinner
from tabprep.utils import p_value_wilcoxon_greater_than_zero, clean_feature_names, clean_series, make_cv_function, p_value_sign_test_median_greater_than_zero, p_value_ttest_greater_than_zero
import time
from category_encoders import LeaveOneOutEncoder
from tabprep.base_preprocessor import BasePreprocessor

class FeatureTypeDetector(BasePreprocessor):
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
                alpha=0.1,
                significance_method='wilcoxon', # ['ttest', 'wilcoxon', 'median]
                verbose=True,
                assign_numeric=False, 

                # Trivial feature test hyperparameters
                detect_numeric_in_string=False, 
                # TODO: Add a hyperparameter to optionally reassess convertible categoricals

                # Interpolation test hyperparameters
                max_degree=3, # max degree of polynomial interpolation
                interpolation_criterion="match", # ['win', 'match']

                # Combination test hyperparameters
                combination_criterion='win',
                combination_test_min_bins=16,
                combination_test_max_bins=2048, #TODO: Test with 90% of unique values as max
                assign_numeric_with_combination=False,
                binning_strategy='lgb', # ['lgb', 'KMeans', 'DT']

                # Multivariate performance test hyperparameters
                mvp_use_data='all', # ['all', 'numeric'] - if 'all', use all columns, if 'numeric', use only numeric columns
                mvp_criterion='significance', # ['significance', 'average'] - if 'significance', use p-value, else just the average score
                mvp_max_cols_use=100, # Maximum number of columns to use in the multivariate performance test
                # Mode test hyperparameters
                max_modes=5,

                # Interaction test hyperparameters
                interaction_mode='random', # ['high_corr', 'all' int]
                n_interaction_candidates=1, # Number of interaction candidates to be generated per feature

                 # Performance test hyperparameters
                 lgb_model_type="unique-based-binned", 
                 fit_cat_models=True,

                 # experimental
                 drop_modes=0,
                 drop_unique=False,
                 ):
        # NOTE: Idea: Reinvent interaction test by first categorizing a dataset into 1) there are already clear categorical features; 2) there are only numerical features; 3) there are only numerical features, but many categorical candidates. And based on that use different interaction tests.
        # TODO: Make sure that everything is deterministic given a fixed seed
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        # Parameters
        self.tests_to_run = tests_to_run
        self.min_q_as_num = min_q_as_num
        self.lgb_model_type = lgb_model_type
        self.max_degree = max_degree
        self.max_modes = max_modes
        self.interpolation_criterion = interpolation_criterion
        self.assign_numeric = assign_numeric
        self.assign_numeric_with_combination = assign_numeric_with_combination
        self.detect_numeric_in_string = detect_numeric_in_string
        self.fit_cat_models = fit_cat_models
        self.combination_test_min_bins = combination_test_min_bins
        self.combination_test_max_bins = combination_test_max_bins
        self.binning_strategy = binning_strategy
        self.combination_criterion = combination_criterion
        self.drop_unique = drop_unique
        self.interaction_mode = interaction_mode
        self.mvp_use_data = mvp_use_data
        # self.mvp_criterion = mvp_criterion
        # self.mvp_max_cols_use = mvp_max_cols_use
        self.n_interaction_candidates = n_interaction_candidates
        self.drop_modes = drop_modes

        # Currently dummy-mean test is necessary as all other tests build on it
        if 'dummy_mean' not in self.tests_to_run:
            self.tests_to_run.insert(0, 'dummy_mean')

        # Functions
        self.cv_func = make_cv_function(target_type=self.target_type, early_stopping_rounds=20, vectorized=False, drop_modes=self.drop_modes)

        # Output variables
        self.orig_dtypes = {}
        self.dtypes = {}
        self.col_names = []
        self.reassigned_features = []
        self.cat_dtype_maps = {}
        self.removed_after_tests = {}
        # TODO: Implement option to drop duplicates
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

    def dummy_mean_test(self, X, y):
        X_use = X.copy()
        self.get_dummy_mean_scores(X_use, y)
        irrelevant_cols = []
        for col in X_use.columns:
            if self.significances[col]["test_irrelevant_mean"]>self.alpha: #'mean equal or worse than dummy'
                self.dtypes[col] = "irrelevant"
                irrelevant_cols.append(col)
        
        if self.verbose:
            print(f"\r{len(irrelevant_cols)}/{len(X_use.columns)} columns are numeric acc. to dummy_mean test.")

        return [col for col in X_use.columns if col not in irrelevant_cols]

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
                print(f"\r{len(interpol_cols)}/{len(remaining_cols)} columns are numeric acc. to {interpolation_method} interpolation test.")

            remaining_cols = [x for x in remaining_cols if x not in interpol_cols]

        return remaining_cols
    
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
                self.scores[col][f"combination_test_{m_bin}"] = self.single_combination_test(x_use, y, max_bin=m_bin, binning_strategy=self.binning_strategy)
                
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
                print(f"\r{len(comb_cols)}/{len(remaining_cols)} columns are numeric acc. to combination test with {m_bin} bins.")


            remaining_cols = [x for x in remaining_cols if x not in comb_cols]

        return remaining_cols
    
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
            self.scores[col]["lgb"] = self.cv_func(clean_feature_names(x_use), y, pipe)

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
                self.scores[col]["lgb-cat"] = self.cv_func(clean_feature_names(x_use).astype("category"), y, pipe)

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

            # TODO: Add significance test for LOO test
            loo_pred = LeaveOneOutEncoder().fit_transform(X[col].astype('category'), y)[col]
            # TODO: Check whether the values are correctly matched for binary targets
            self.loo_scores[col] = metric(y, loo_pred)

            if self.loo_scores[col] < dummy_score:
                self.dtypes[col] = "numeric"
                loo_cols.append(col)

        if self.verbose:
            print(f"\r{len(loo_cols)}/{len(X.columns)} columns are numeric acc. to Leave-One-Out test.")

        remaining_cols = [x for x in X.columns if x not in loo_cols]

        return remaining_cols
    
    def get_interaction_candidates_bottomup(self, 
                                        x, X_num, y, 
                                        col_method='random', # ['random', 'target_corr', 'feature_corr']
                                        interaction_method='random', # ['random', 'correlation']
                                        n_candidates=1, replace=False,
                                        seed=42
                                        ):
        np.random.seed(seed)
        if col_method == 'random':
            cols_to_use = np.random.choice(X_num.columns, np.min([len(X_num.columns),n_candidates]), replace=replace)
        elif col_method == 'target_corr':
            cols_to_use = X_num.corrwith(y).abs().sort_values(ascending=False).index[:n_candidates]
        elif col_method == 'feature_corr':
            cols_to_use = X_num.corrwith(x).abs().sort_values(ascending=False).index[:n_candidates]
        else:
            raise ValueError(f"Unknown col_method: {col_method}")

        X_int = pd.DataFrame()
        if interaction_method == 'random':
            interactions = np.random.choice(["x","/","+","-"], np.min([len(X_num.columns),n_candidates]), replace=True)
            for col, interaction in zip(cols_to_use, interactions):
                if interaction == "/":
                    X_int[f"{x.name}_{interaction}_{col}"] = (x / X_num[col].replace(0, np.nan)).values
                elif interaction == "x":
                    X_int[f"{x.name}_{interaction}_{col}"] = (x * X_num[col]).values
                elif interaction == "-":
                    X_int[f"{x.name}_{interaction}_{col}"] = (x - X_num[col]).values
                elif interaction == "+":
                    X_int[f"{x.name}_{interaction}_{col}"] = (x + X_num[col]).values
        elif interaction_method == 'correlation':
            X_int = pd.DataFrame(
                index=X_num.index, 
                columns=[f"{x.name}_/_{col2}" for col2 in cols_to_use]+ \
                [f"{x.name}_x_{col2}" for col2 in cols_to_use]+ \
                [f"{x.name}_-_{col2}" for col2 in cols_to_use]+ \
                [f"{x.name}_+_{col2}" for col2 in cols_to_use]
            )
            X_int[[f"{x.name}_/_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) / X_num[cols_to_use].replace(0, np.nan).values).values
            X_int[[f"{x.name}_x_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) * X_num[cols_to_use].values).values
            X_int[[f"{x.name}_-_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) - X_num[cols_to_use].values).values
            X_int[[f"{x.name}_+_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) + X_num[cols_to_use].values).values

            # Filter weird cases
            X_int = X_int.loc[:, X_int.nunique()>2] 

            cors = X_int.corrwith(y, method='spearman').abs()
            final_cols = [cors[[interaction for interaction in cors.index if col in interaction]].idxmax() for col in cols_to_use]

            X_int = X_int[final_cols]
        else:
            raise ValueError(f"Unknown interaction_method: {interaction_method}")
        
        return X_int

    def get_interaction_candidates_full(self, 
                                        x, X_num, y, 
                                        method='target_corr', # ['target_corr', 'corr_improve_over_base']
                                        n_candidates=1, 
                                        seed=42
                                        ):
        np.random.seed(seed)
        ### 1. Get Interactions
        cols_to_use = [c for c in X_num.columns if c != x.name]
        
        X_int = pd.DataFrame(
            index=X_num.index, 
            columns=[f"{x.name}_/_{col2}" for col2 in cols_to_use]+ \
            [f"{x.name}_x_{col2}" for col2 in cols_to_use]+ \
            [f"{x.name}_-_{col2}" for col2 in cols_to_use]+ \
            [f"{x.name}_+_{col2}" for col2 in cols_to_use]
        )
        
        # TODO: Think about whether reversing - and / makes sense
        # TODO: Think about whether setting nans for division by zero is appropriate
        X_int[[f"{x.name}_/_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) / X_num[cols_to_use].replace(0, np.nan).values).values
        X_int[[f"{x.name}_x_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) * X_num[cols_to_use].values).values
        X_int[[f"{x.name}_-_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) - X_num[cols_to_use].values).values
        X_int[[f"{x.name}_+_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) + X_num[cols_to_use].values).values

        # Filter weird cases
        X_int = X_int.loc[:, X_int.nunique()>2]
        
        corr_X_int = X_int.corrwith(y, method='spearman').abs().sort_values(ascending=False)

        if method == 'target_corr':
            candidate_cols = corr_X_int.index[:n_candidates].tolist()
        # EXPERIMENTAL;DOESNT WORK CURRENTLY
        # elif method == 'corr_improve_over_base':
        #     base_cors = X_num.corrwith(y).abs()
        #     int_cors = corr_X_int.to_frame()
        #     int_cors_diff = int_cors.apply(lambda x: float([x-base_cors.loc[x.name.split(f"_{i}_")].max() for i in ['+', '-', '/', 'x'] if len(x.name.split(f"_{i}_"))>1][0][0]),axis=1)
        #     candidate_cols = int_cors_diff.sort_values(ascending=False).index[:1].tolist()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'target_corr' or 'corr_improve_over_base'.")
        
        return X_int[candidate_cols]

    def get_interaction_candidates(self, x, X_num, y, interaction_mode='random', n_interaction_candidates=1):
        if interaction_mode=='full':
            X_int = self.get_interaction_candidates_full(x, X_num, y, method='target_corr', n_candidates=n_interaction_candidates)
        elif interaction_mode=='random':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='random', interaction_method='random',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='target':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='target_corr', interaction_method='random',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='feature':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='feature_corr', interaction_method='random',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='random-best':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='random', interaction_method='correlation',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='target-best':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='target_corr', interaction_method='correlation',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='feature-best':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='feature_corr', interaction_method='correlation',
                                                        n_candidates=n_interaction_candidates, replace=False)
        else:
            raise ValueError(f"Unknown interaction_mode: {interaction_mode}. Use 'full', 'random', 'target', 'feature', 'random-best', 'target-best', or 'feature-best'.")
        
        return X_int
    
    def interaction_test(self, X_cand, y, X_num):
        '''
        NOTE: This test is WIP and currently not used in the default pipeline.
        Three aspects matter:
        1. Does the new feature behaves numerically? - are the interpolation and combination tests positive?
        2. Does the feature improve performance over the base features at all?
        3. Is the feature interaction truly numerical or just the same as a combination of two base features?
        Therefore, we label a feature as numerical if its interaction 
            a) behaves numerically,
            b) improves performance, and 
            c) is not just a combination of the base features. 
        '''
       
        interaction_cols = []
        for cnum, col in enumerate(X_cand.columns):
            if self.verbose:
                print(f"\rInteraction test for column {cnum+1}/{len(X_cand.columns)} columns processed", end="", flush=True)

            # 1. Get interaction candidates
            X_int = self.get_interaction_candidates(X_cand[col], X_num, y,
                                                    interaction_mode=self.interaction_mode,
                                                    n_interaction_candidates=self.n_interaction_candidates)
   
            for highest_corr in X_int.columns:
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
                self.dummy_mean_test(X_use, y)
                self.significances[col][f"test_arithmetic-mean_superior_single-best"] = self.significance_test(
                    self.scores[arithmetic_col]['mean'] - self.scores[stronger_col][stronger_col_setting]
                )       
                if self.significances[col][f"test_arithmetic-mean_superior_single-best"]<self.alpha:
                    arithmetic_improves_performance = True
                else:
                    arithmetic_improves_performance = False

                ### Combination test
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
                

                # 2. CHECK - The arithmetic combination feature improves performance over the base features
                self.significances[col][f"test_arithmetic-best_superior_single-best"] = self.significance_test(
                    self.scores[arithmetic_col][best_setting] - self.scores[stronger_col][stronger_col_setting]
                )       
                if self.significances[col][f"test_arithmetic-best_superior_single-best"]<self.alpha:
                    arithmetic_improves_performance = True

                # 3. CHECK - The arithmetic combination feature is not just a combination of the base features

                if arithmetic_is_numeric and arithmetic_improves_performance:
                    self.dtypes[col] = "numeric"
                    interaction_cols.append(col)
                    break


                ### 5. Test whether categorical interaction is better than numerical interaction
                # Combine interaction performance
                # x = X_full[col].astype(str) + X_full[col2].astype(str)
                # if self.target_type == "regression":
                #     self.scores[arithmetic_col]['combine'] = self.cv_func(x.to_frame(), y, Pipeline(steps=[('model', TargetMeanRegressor())]))
                # else:
                #     self.scores[arithmetic_col]['combine'] = self.cv_func(x.to_frame(), y, Pipeline(steps=[('model', TargetMeanClassifier())]))
                

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
            print(f"\rMode test: {cnum+1}/{len(X.columns)} columns processed", end="", flush=True)
            x = X[col]
            modes = x.value_counts(dropna=False)
            for num_modes in range(1,self.max_modes+1):
                mode_map = dict(zip(modes.index[:num_modes], range(num_modes)))
                x_use = x.map(mode_map).fillna(num_modes)
                # roc_auc_score(y, TargetMeanClassifier().fit(X[[x.name]]==0, y).predict_proba(X[[x.name]]==0)[:,1])
                self.scores[col][f'mode{num_modes}'] = self.cv_func(x_use.to_frame(), y, Pipeline([('model',  target_model)]))

                self.significances[col][f"{num_modes}_superior_mean"] = self.significance_test(
                    self.scores[col][f'mode{num_modes}'] - self.scores[col]['mean']
                )

                if self.significances[col][f"{num_modes}_superior_mean"]<self.alpha:
                    self.dtypes[col] = "multimodal"
                    mode_cols.append(col)
                    break

        remaining_cols = [x for x in X.columns if x not in mode_cols]

        return remaining_cols         

    # def multivariate_performance_test(self, X_in, y_in, test_cols, max_cols_use=100):
    #     # TODO: Implement a small TabM model as a proxy for neural nets as the downstream model
    #     # TODO: Implement a method to process features in chunks if we have more than x columns to test. Maybe something like the cat-interaction test can be used to determine which columns to process together.
    #     np.random.seed(42)
    #     X = X_in.copy()
    #     y = y_in.copy()
        
    #     if X.drop(test_cols, axis=1).shape[1] > max_cols_use:
    #         X = pd.concat([
    #             X[np.random.choice(X.drop(test_cols, axis=1).columns, size=max_cols_use, replace=False)],
    #             X[test_cols]
    #         ], axis=1)

    #         prefix = f"LGB-subsample{X.shape[1]}"
    #     else:
    #         prefix = "LGB-full"

    #     params = {
    #                 "objective": "binary" if self.target_type=="binary" else "regression",
    #                 "boosting_type": "gbdt",
    #                 "n_estimators": 1000,
    #                 'min_samples_leaf': 2,
    #                 "max_depth": 5,
    #                 "verbosity": -1
    #             }
        
        
    #     self.scores[prefix] = {}
    #     self.significances[prefix] = {}
    #     if self.verbose:
    #         print(f"\rMultivariate performance test with {len(test_cols)} columns.")
    #     # Get performance for all columns as numerical columns
    #     X_use = X.copy()
    #     # TODO: Think about including AG preprocessor prior to model fitting
    #     obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
    #     X_use[obj_cols] = X_use[obj_cols].astype('category')
    #     X_use[test_cols] = X_use[test_cols].astype(float)  # Use float dtype for numerical columns
    #     model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
    #     pipe = Pipeline([("model", model)])
    #     self.scores[prefix]['raw'], all_num_iter = self.cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)
        
    #     X_use = X.copy() 
    #     # X_use = X_use.astype('float')  # Use category dtype for categorical columns
    #     obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
    #     X_use[obj_cols] = X_use[obj_cols].astype('category')
    #     X_use[test_cols] = X_use[test_cols].astype('category')  # Use category dtype for categorical columns
    #     model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
    #     pipe = Pipeline([("model", model)])
    #     self.scores[prefix][f"{len(test_cols)}-CAT"], all_remcat_iter = self.cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)

    #     self.significances[prefix][f"{len(test_cols)}-CAT"] = p_value_wilcoxon_greater_than_zero(self.scores[prefix][f"{len(test_cols)}-CAT"]-self.scores[prefix]['raw'])
        
    #     if self.mvp_criterion=='significance':
    #         significant = self.significances[prefix][f"{len(test_cols)}-CAT"] < self.alpha
    #     elif self.mvp_criterion=='average':
    #         significant = np.mean(self.scores[prefix][f"{len(test_cols)}-CAT"] - self.scores[prefix]['raw']) > 0
    #     else:
    #         raise ValueError(f"Unknown mvp_criterion: {self.mvp_criterion}. Use 'significance' or 'average'.")
        
    #     if significant:
    #         if self.verbose:
    #             print(f"ALL-CAT is significantly better than ALL-NUM with p-value {self.significances[prefix][f'{len(test_cols)}-CAT']:.3f}")
    #             print(f"Execute per-column tests in backward selection mode.")
    #         for col in test_cols:
    #             self.dtypes[col] = 'categorical'
    #         selection_mode = 'backward'
    #     else:
    #         if self.verbose:
    #             print(f"ALL-CAT is not significantly better than ALL-NUM with p-value {self.significances[prefix][f'{len(test_cols)}-CAT']:.3f}")
    #             print(f"Execute per-column tests in forward selection mode.")
    #         for col in test_cols:
    #             self.dtypes[col] = 'numeric'
    #         selection_mode = 'forward'

    #     if len(test_cols) > 1:
    #         cat_cols = []
    #         for num, col in enumerate(test_cols):
    #             if self.verbose:
    #                 print(f"\rColumn {num+1}/{len(test_cols)}: {col}", end="", flush=True)
    #             X_use = X.copy()
    #             obj_cols = X_use.select_dtypes(include=['object']).columns.tolist()
    #             X_use[obj_cols] = X_use[obj_cols].astype('category')
    #             if selection_mode == 'backward':
    #                 use_scores = self.scores[prefix][f"{len(test_cols)}-CAT"]
    #                 X_use[test_cols] = X_use[test_cols].astype('category')  # Use category dtype for categorical columns
    #                 X_use[col] = X_use[col].astype(float)  
    #                 suffix = "-asnum"
    #             else:
    #                 use_scores = self.scores[prefix]['raw']
    #                 X_use[test_cols] = X_use[test_cols].astype(float)  # Use category dtype for categorical columns
    #                 X_use[col] = X_use[col].astype('category')
    #                 suffix = "-ascat"
    #             # TODO: Make sure to prevent leaks with category dtypes
    #             model = lgb.LGBMClassifier(**params) if self.target_type=="binary" else lgb.LGBMRegressor(**params)
    #             pipe = Pipeline([("model", model)])
    #             self.scores[prefix][f"{col}{suffix}"], cat_iter = self.cv_func(clean_feature_names(X_use), y, pipe, return_iterations=True)

    #             self.significances[prefix][f"{col}{suffix}_superior_{prefix}"] = p_value_wilcoxon_greater_than_zero(self.scores[prefix][f"{col}{suffix}"]-use_scores)

    #             if self.mvp_criterion=='significance':
    #                 significant = self.significances[prefix][f"{col}{suffix}_superior_{prefix}"] < self.alpha
    #             elif self.mvp_criterion=='average':
    #                 significant = np.mean(self.scores[prefix][f"{col}{suffix}"]-use_scores) > 0
    #             else:
    #                 raise ValueError(f"Unknown mvp_criterion: {self.mvp_criterion}. Use 'significance' or 'average'.")

    #             # if significant:
    #             #     if self.verbose:
    #             #         print(f"{col}{suffix} is significantly better in {selection_mode} mode with p-value {self.significances[prefix][f'{col}{suffix}_superior_{prefix}']:.3f}")
    #             #     if self.verbose:
    #             #         print(f"{col}{suffix} is significantly better in {selection_mode} mode with p-value {self.significances[prefix][f'{col}{suffix}_superior_{prefix}']:.3f}")
    #                 self.dtypes[col] = 'numeric' if selection_mode == 'backward' else 'categorical'
    #             if self.dtypes[col] == 'categorical':
    #                 cat_cols.append(col)
    #         if self.verbose:
    #             print(f"\r{len(cat_cols)}/{len(test_cols)} columns are categorical.")
    #     elif self.verbose:
    #         if self.dtypes[test_cols[0]] == 'categorical':
    #             print(f"\r1/{len(test_cols)} columns are categorical.")
    #     return []


    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                X_out[test_cols] = X_out[test_cols].astype(float)
            elif mode == 'forward':
                X_out[test_cols] = X_out[test_cols].astype('category')
        else:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                X_out[col] = X_out[col].astype(float)
            elif mode == 'forward':
                X_out[col] = X_out[col].astype('category')
        
        return X_out
        
    def multivariate_performance_test(self, X_cand_in, y_in, 
                                      test_cols, max_cols_use=100):
        suffix = 'CAT'
        cat_cols = super().multivariate_performance_test(X_cand_in, y_in, 
                                      test_cols, suffix=suffix,  max_cols_use=max_cols_use)
        
        for col in test_cols:
            if col in cat_cols:
                self.dtypes[col] = 'categorical'
            else:
                self.dtypes[col] = 'numeric'
        if self.verbose:
            if self.dtypes[test_cols[0]] == 'categorical':
                print(f"\r1/{len(test_cols)} columns are categorical.")
        
        return []

    def run_test(self, X, y, test_name, test_cols=list(), **kwargs):
        start = time.time()
        if test_name == 'dummy_mean':
            remaining_cols = self.dummy_mean_test(X, y)
        elif test_name == 'leave_one_out':
            remaining_cols = self.leave_one_out_test(X, y)
        elif test_name == 'combination':
            remaining_cols = self.combination_test(X, y, verbose=self.verbose)
        elif test_name == 'interpolation':
            remaining_cols = self.interpolation_test(X, y, max_degree=self.max_degree)
        elif test_name == 'interpolation-log':
            remaining_cols = self.interpolation_test(np.log(X+1e-5), y, max_degree=self.max_degree)
        elif test_name == 'interaction':
            if 'X_num' in kwargs:
                X_num = kwargs['X_num']
            else:
                X_num = X.copy()
            remaining_cols = self.interaction_test(X, y, X_num=X_num)
        elif test_name == 'performance':
            remaining_cols = self.performance_test(X, y)
        elif test_name == 'mode':
            remaining_cols = self.mode_test(X, y)
        elif test_name == 'multivariate_performance':
            remaining_cols = self.multivariate_performance_test(X, y, test_cols, max_cols_use=self.mvp_max_cols_use)
        else:
            print(f"Unknown test_name: {test_name}. Use 'dummy_mean', 'leave_one_out', 'combination', 'interpolation', 'interaction' or 'performance'.")
            remaining_cols = X.columns.values.tolist()

        self.times[test_name] = time.time() - start
        if test_name == 'multivariate_performance':
            for col in test_cols:
                self.removed_after_tests[col] = test_name
        else:
            for col in set(X.columns) - set(remaining_cols):
                self.removed_after_tests[col] = test_name
            # if self.verbose:
            #     print(f"\r{len(remaining_cols)/len(X.columns):.2%} of the candidate columns remain after the {test_name} test.")
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

        self.original_cat_features = [key for key, value in self.orig_dtypes.items() if value == "categorical"]

        if self.target_type=="multiclass":
            # TODO: Fix this hack
            y = (y==y.value_counts().index[0]).astype(int)  # make it binary
            self.target_type = "binary"
        elif self.target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
            y = (y==y.value_counts().index[0]).astype(int)  # make it numeric
        else:
            y = y.astype(float)

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
        X_cand = X[numeric_cand_cols].copy().astype(float)
        X_full = X_cand.copy()
        # TODO: Reinvent the whole 'assign_numeric' logic

        ### TEST NUMERIC CANDIDATE COLUMNS
        for test_name in self.tests_to_run:            
            if test_name == 'interaction':
                numeric_cols = [key for key,value in self.dtypes.items() if value=='numeric']
                if len(numeric_cols)>0:
                    X_num = X_full[numeric_cols]
                elif X_cand.shape[1]>1:
                    X_num = X_cand.copy()
                elif X_full.drop(X_cand.columns,axis=1).shape[1]>1:
                    possible_cols = X_full.drop(X_cand.columns,axis=1).columns
                    X_num = X_full[np.random.choice(possible_cols, min([self.n_interaction_candidates, len(possible_cols)]), replace=False)].copy()
            else:
                X_num = None

            if test_name == 'multivariate_performance':
                if self.assign_numeric:
                    test_cols = numeric_cand_cols
                else:
                    test_cols = list(set(numeric_cand_cols)-set(self.original_cat_features))
                    if len(test_cols)==0:
                        numeric_cand_cols = []
                        continue
                if self.mvp_use_data=='all':
                    numeric_cand_cols = self.run_test(X, y, test_name=test_name, test_cols=test_cols)
                elif self.mvp_use_data=='numeric':
                    numeric_cand_cols = self.run_test(X_full, y, test_name=test_name, test_cols=test_cols)
            else:
                numeric_cand_cols = self.run_test(X_cand, y, test_name, X_num=X_num)
            if len(numeric_cand_cols)==0:
                continue
            else:
                X_cand = X_cand[numeric_cand_cols].copy()

        # Simply assign all remaining numeric candidate columns as categorical
        for col in numeric_cand_cols:
            self.dtypes[col] = "categorical"

        # TODO: Double-check that the assignment of categorical (&numerical) features is correct
        # Prepare objects to transform columns
        self.reassign_cat_cols = [col for col in X.columns if self.dtypes[col]=="categorical" and self.orig_dtypes[col]!="categorical"]
        for col in self.reassign_cat_cols:
            self.cat_dtype_maps[col] = pd.CategoricalDtype(categories=list(X[col].astype(str).fillna("nan").unique()))

        if self.assign_numeric:
            self.reassign_num_cols = [col for col in X.columns if self.dtypes[col]=="numeric" and self.orig_dtypes[col]!="numeric"]
            self.numeric_means = {col: X_full[col].mean() for col in self.reassign_num_cols}

        return self

    def transform(self, X_input, mode = "overwrite", fillna_numeric=False):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)

        X = X_input.copy()

        for col in self.cat_dtype_maps:
            if mode == "overwrite":
                X[col] = X[col].astype(str).fillna("nan").astype(self.cat_dtype_maps[col])
            elif mode == "add":
                X[col+'_cat'] = X[col].astype(str).fillna("nan").astype(self.cat_dtype_maps[col])
                self.dtypes[col+'_cat'] = "categorical"
        
        if self.assign_numeric:
            for col in self.reassign_num_cols:
                X[col] = pd.to_numeric(X[col], errors= 'coerce').astype(float)
                if X[col].isna().any() and fillna_numeric:
                    X[col] = X[col].fillna(self.numeric_means[col])

        return X

    def get_params(self, deep=True):
        return {
            "target_type": self.target_type,
            "tests_to_run": self.tests_to_run,
            "min_q_as_num": self.min_q_as_num,
            "n_folds": self.n_folds,
            "lgb_model_type": self.lgb_model_type,
            "max_degree": self.max_degree,
            "max_modes": self.max_modes,
            "interpolation_criterion": self.interpolation_criterion,
            "assign_numeric": self.assign_numeric,
            "assign_numeric_with_combination": self.assign_numeric_with_combination,
            "detect_numeric_in_string": self.detect_numeric_in_string,
            "fit_cat_models": self.fit_cat_models,
            "combination_test_min_bins": self.combination_test_min_bins,
            "combination_test_max_bins": self.combination_test_max_bins,
            "binning_strategy": self.binning_strategy,
            "combination_criterion": self.combination_criterion,
            "drop_unique": self.drop_unique,
            "alpha": self.alpha,
            "interaction_mode": self.interaction_mode,
            "mvp_use_data": self.mvp_use_data,
            "mvp_criterion": self.mvp_criterion,
            "mvp_max_cols_use": self.mvp_max_cols_use,
            "n_interaction_candidates": self.n_interaction_candidates,
            "drop_modes": self.drop_modes,
            "significance_method": self.significance_method,
            "verbose": self.verbose
            
        }

