import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, log_loss
from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor, UnivariateLinearRegressor, UnivariateLogisticClassifier, PolynomialLogisticClassifier, PolynomialRegressor, UnivariateThresholdClassifier, MultiFeatureTargetMeanClassifier, MultiFeatureUnivariateLogisticClassifier, LightGBMBinner, KMeansBinner, TargetMeanRegressorCut, TargetMeanClassifierCut
from tabprep.utils import p_value_wilcoxon_greater_than_zero, clean_feature_names, clean_series, make_cv_function, p_value_sign_test_median_greater_than_zero, p_value_ttest_greater_than_zero
import time
from category_encoders import LeaveOneOutEncoder
from tabprep.base_preprocessor import BasePreprocessor

from tabprep.preprocessors import TargetRepresenter, FreqAdder, CatIntAdder, CatGroupByAdder
from tabprep.num_interaction import NumericalInteractionDetector
from tabprep.groupby_interactions import GroupByFeatureEngineer
from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator

class AllInOneEngineer(BasePreprocessor):
    def __init__(self, target_type, use_residuals=False,
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                 combination_test_min_bins=2, combination_test_max_bins=2048, binning_strategy='lgb',                 
                 engineering_techniques: list=[],
                 ):
        self.original_target_type = target_type
        # NOTE: Idea: Reinvent interaction test by first categorizing a dataset into 1) there are already clear categorical features; 2) there are only numerical features; 3) there are only numerical features, but many categorical candidates. And based on that use different interaction tests.
        # TODO: Make sure that everything is deterministic given a fixed seed
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.use_residuals = use_residuals
        self.combination_test_min_bins = combination_test_min_bins
        self.combination_test_max_bins = combination_test_max_bins
        self.binning_strategy = binning_strategy
        self.engineering_techniques = engineering_techniques

        # Functions
        self.cv_func = make_cv_function(target_type=self.target_type, early_stopping_rounds=20)
        if self.target_type=='binary':
            self.scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
        elif self.target_type=='regression':
            self.scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
        elif self.target_type=='multiclass':
            self.scorer = lambda ytr, ypr: -log_loss(ytr, ypr) # r2_score

        # Output variables
        self.scores = {}
        self.significances = {}
        self.feature_target_rep = {}
        self.transformers = []
        self.drop_cols = []

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
                self.scores[col][f"cat_threshold_test_{thresh}"] = self.cv_func(X[[col]], y, pipe)

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


    def fit(self, X_input, y_input=None, verbose=False):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)
        if not isinstance(y_input, pd.DataFrame):
            y_input = pd.Series(y_input)
        X = X_input.copy()
        y = y_input.copy()

        y = self.adjust_target_format(y) # TODO: Adapt for multiclass

        # Save some properties of the original input data
        self.orig_dtypes = {col: "categorical" if dtype in ["object", "category", str] else "numeric" for col,dtype in dict(X.dtypes).items()}
        for col in self.orig_dtypes:
            if X[col].nunique()<=2:
                self.orig_dtypes[col] = "binary"
        self.original_cat_features = [key for key, value in self.orig_dtypes.items() if value == "categorical"]
        self.col_names = X.columns
        self.changes_to_cols = {col: "None" for col in self.col_names}
        self.changes_to_cols = {col: "raw" for col in self.col_names} 

        # 0. Apply AG preprocessing
        old_cols = X.columns.tolist()
        X = DropDuplicatesFeatureGenerator().fit_transform(X)
        if X.shape[1] < len(old_cols):
            self.drop_cols.extend([col for col in old_cols if col not in X.columns])

        # 1. Detect irrelevant features
        # TODO: Add AG method here
        self.scores['full'] = {}
        if 'drop_irrelevant' in self.engineering_techniques:
            self.scores['full']['lgb-irrelevant'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type='irrelevant')
            self.irrelevant_feature_importances = pd.DataFrame(feature_importances).mean().sort_values()

            # TODO: Add as a preprocessor
            self.drop_cols.extend(self.irrelevant_feature_importances[self.irrelevant_feature_importances < self.alpha].index.tolist())

        X = X.drop(columns=self.drop_cols, errors='ignore', axis=1)
        X_bin = X[X.columns[X.nunique()<=2]]
        X_cat = X[X.columns[X.nunique()>2]].select_dtypes(include=['object', 'category'])
        X_num = X[X.columns[X.nunique()>2]].select_dtypes(exclude=['object', 'category'])

        assert X_cat.shape[1] + X_num.shape[1] + X_bin.shape[1] ==  X.shape[1], "Not all features covered through num/cat selection."

        # 2. Detect cat as ordinal
        # if 'cat_ordinal' in self.engineering_techniques:
        #     # 1. Get all dummy-mean scores
        #     self.get_dummy_mean_scores(X_cat, y)
            
        #     # Apply categorical threshold test
        #     cat_threshold_test = self.cat_threshold_test(X_cat, y)

        #     # 2. Get all combination scores
        #     X_cat_num = X_cat.copy()
        #     for col in X_cat_num:
        #         num_convertible = pd.to_numeric(X[col].dropna(), errors='coerce').notna().all()
        #         if num_convertible:
        #             X_cat_num[col] = pd.to_numeric(X_cat_num[col], errors='coerce')
        #         else:
        #             X_cat_num[col] = OrdinalEncoder().fit_transform(X[[col]]).flatten()
                
            
        #     num_perf, _ = self.get_lgb_performance(X_num, y, lgb_model_type='default')
        #     cat_perf, _ = self.get_lgb_performance(X_cat, y, lgb_model_type='default')
        #     numcat_perf, _ = self.get_lgb_performance(X_cat_num, y, lgb_model_type='default')
        #     all_perf, _ =  = self.get_lgb_performance(X, y, lgb_model_type='default')
        #     pd.DataFrame({
        #         'ALL': all_perf,
        #         'NUM': num_perf,
        #         'CAT': cat_perf,
        #         'CAT_as_NUM': numcat_perf
        #         }).mean()

        #     comb_cols = self.combination_test(X_cat_num, y, early_stopping=False)

        #     # 3. Get linear scores
        #     linear_cols = self.interpolation_test(X_cat_num, y, max_degree=1)

        # 2. Categorical frequency
        if 'cat_freq' in self.engineering_techniques:
            if len(X_cat.columns) > 0:
                cat_freq = FreqAdder()
                candidate_cols = cat_freq.filter_candidates_by_distinctiveness(X_cat)
                if len(candidate_cols) > 0:
                    self.scores['full']['lgb-default'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type='default')
                    self.scores['cat_freq'] = {}
                    self.scores['cat_freq']['lgb-default'], preds,feature_importances = self.get_lgb_performance(X, y, custom_prep=[FreqAdder(candidate_cols=candidate_cols)], lgb_model_type='default')

                    if np.mean(self.scores['cat_freq']['lgb-default']) > np.mean(self.scores['full']['lgb-default']):
                        self.transformers.append(FreqAdder(candidate_cols=candidate_cols).fit(X, y))                

        # # 3. Categorical interactions
        if 'cat_int' in self.engineering_techniques:
            lgb_model_use = 'fast'
            if len(X_cat.columns) > 1 and sum(X_cat.nunique() > 5)>2: # TODO: Make this a hyperparameter
                if f'lgb-{lgb_model_use}' not in self.scores['full']:
                    self.scores['full'][f'lgb-{lgb_model_use}'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use)
                if self.use_residuals:
                    residuals = y-pd.concat(full_preds).sort_index()
                curr_comp = 'full'
                best_order = 1
                use_freq = {}
                for order in range(2,7): # TODO: Add max_order as parameter
                    if len(X_cat.columns) >= order:
                        self.scores[f'cat_int{order}'] = {}
                        self.scores[f'cat_int{order}_andFreq'] = {}
                        # self.scores[f'cat_int{order}_onlyFreq'] = {}

                        if self.use_residuals:
                            self.scores[f'cat_int{order}'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, use_filters=False)], lgb_model_type=lgb_model_use, residuals=residuals)
                            if target_type != 'regression':
                                self.scores[f'cat_int{order}'][f'lgb-{lgb_model_use}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])                    
                            # TODO: Add some additional test to not always attempt +freq
                            # I.e., only if the previous order was better with freq
                            self.scores[f'cat_int{order}_andFreq'][f'lgb-{lgb_model_use}'], intfreq_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, add_freq=True, use_filters=False)], lgb_model_type=lgb_model_use, residuals=residuals)
                            if target_type != 'regression':
                                self.scores[f'cat_int{order}_andFreq'][f'lgb-{lgb_model_use}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in intfreq_preds])
                        else:
                            self.scores[f'cat_int{order}'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, use_filters=False)], lgb_model_type=lgb_model_use)
                            self.scores[f'cat_int{order}_andFreq'][f'lgb-{lgb_model_use}'], intfreq_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, add_freq=True, use_filters=False)], lgb_model_type=lgb_model_use)
                        
                        andFreq_beats_int = np.mean(self.scores[f'cat_int{order}_andFreq'][f'lgb-{lgb_model_use}']) > np.mean(self.scores[f'cat_int{order}'][f'lgb-{lgb_model_use}'])
                        andFreq_beats_base = np.mean(self.scores[f'cat_int{order}_andFreq'][f'lgb-{lgb_model_use}']) > np.mean(self.scores[curr_comp][f'lgb-{lgb_model_use}'])
                        int_beats_base = np.mean(self.scores[f'cat_int{order}'][f'lgb-{lgb_model_use}']) > np.mean(self.scores[curr_comp][f'lgb-{lgb_model_use}'])

                        if andFreq_beats_int and andFreq_beats_base:
                            curr_comp = f'cat_int{order}_andFreq'
                            best_order = order
                            use_freq[order] = True
                            residuals = y - pd.concat(intfreq_preds).sort_index()
                        elif int_beats_base:
                            curr_comp = f'cat_int{order}'
                            best_order = order
                            use_freq[order] = False
                            residuals = y - pd.concat(int_preds).sort_index()
                        else:
                            break
                if best_order > 1:
                    self.transformers.append(CatIntAdder(self.target_type, max_order=best_order, add_freq=use_freq[best_order], use_filters=False).fit(X, y))

        # 4. Categorical GroupBy interactions
        if 'cat_groupby' in self.engineering_techniques:
            lgb_model_use = 'fast'
            if len(X_cat.columns) > 1 and sum(X_cat.nunique() > 5)>2: # TODO: Test to filter low-cardinality for groupby
                if f'lgb-{lgb_model_use}' not in self.scores['full']:
                    self.scores['full'][f'lgb-{lgb_model_use}'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use)
                if self.use_residuals:
                    residuals = y-pd.concat(full_preds).sort_index()
                
                self.scores[f'cat_groupby'] = {}
                if self.use_residuals:
                    self.scores[f'cat_groupby'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatGroupByAdder()], lgb_model_type=lgb_model_use, residuals=residuals)
                    if target_type != 'regression':
                        self.scores[f'cat_groupby'][f'lgb-{lgb_model_use}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])                    
                else:
                    self.scores[f'cat_groupby'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatGroupByAdder()], lgb_model_type=lgb_model_use)

                if np.mean(self.scores[f'cat_groupby'][f'lgb-{lgb_model_use}']) > np.mean(self.scores['full'][f'lgb-{lgb_model_use}']):
                    self.transformers.append(CatGroupByAdder().fit(X, y))
                print("!")

        # 5. Numerical interactions
        if 'num_int' in self.engineering_techniques:
            lgb_model_use = 'default'
            if len(X_num.columns) > 1 and sum(X_num.nunique() > 5)>2: # TODO: Make this a hyperparameter
                if f'lgb-{lgb_model_use}' not in self.scores['full']:
                    self.scores['full'][f'lgb-{lgb_model_use}'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use)
                if self.use_residuals:
                    residuals = y-pd.concat(full_preds).sort_index()
                curr_comp = 'full'
            
                
                imp_df = pd.DataFrame(feature_importances).mean()
                candidate_cols = imp_df.loc[X_num.columns].sort_values(ascending=False).head(20).index.tolist()
                
                prep_args = {}
                best_order = 1
                for order in range(2,3): # TODO: Add max_order as parameter
                    if len(X_num.columns) >= order:
                        self.scores[f'num_int{order}'] = {}
                        
                        prep_args[order] = {
                            'target_type': self.target_type, 
                            'max_order': order, 
                            'num_operations': 'all',
                            'use_mvp': False,
                            'corr_thresh': .95,
                            'select_n_candidates': 500,
                            'apply_filters': False,
                            'candidate_cols': candidate_cols
                        }

                        preprocessor = NumericalInteractionDetector(
                            **prep_args[order]
                        )
                        
                        if self.use_residuals:
                            self.scores[f'num_int{order}'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=lgb_model_use, residuals=residuals)
                            if target_type != 'regression':
                                self.scores[f'num_int{order}'][f'lgb-{lgb_model_use}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])                    
                        else:
                            self.scores[f'num_int{order}'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=lgb_model_use)
                        
                        if np.mean(self.scores[f'num_int{order}'][f'lgb-{lgb_model_use}']) > np.mean(self.scores[curr_comp][f'lgb-{lgb_model_use}']):
                            curr_comp = f'num_int{order}'
                            best_order = order
                            if self.use_residuals:
                                residuals = y - pd.concat(int_preds).sort_index()
                        else:
                            break

                if best_order > 1:
                    self.transformers.append(
                        NumericalInteractionDetector(
                            **prep_args[best_order]
                        ).fit(X, y)
                        )
        
        # 6. Cat-by-NUM GroupBy interactions
        if 'groupby' in self.engineering_techniques:
            lgb_model_use = 'default'
            if len(X_cat.columns) > 0 and any(X_cat.nunique() > 5) and len(X_num.columns) > 0 and any(X_num.nunique() > 5): # TODO: Test to filter low-cardinality for groupby
                if f'lgb-{lgb_model_use}' not in self.scores['full']:
                    self.scores['full'][f'lgb-{lgb_model_use}'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use)
                if self.use_residuals:
                    residuals = y-pd.concat(full_preds).sort_index()
                
                prep_params = {
                    'target_type': self.target_type,
                    'min_cardinality': 6,
                    'use_mvp': False,
                    'mean_difference': True,
                    'num_as_cat': False,
                }

                preprocessor = GroupByFeatureEngineer(
                    **prep_params
                )

                self.scores[f'groupby'] = {}
                if self.use_residuals:
                    self.scores[f'groupby'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=lgb_model_use, residuals=residuals)
                    if target_type != 'regression':
                        self.scores[f'groupby'][f'lgb-{lgb_model_use}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])                    
                else:
                    self.scores[f'groupby'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=lgb_model_use)

                if np.mean(self.scores[f'groupby'][f'lgb-{lgb_model_use}']) > np.mean(self.scores['full'][f'lgb-{lgb_model_use}']):
                    self.transformers.append(GroupByFeatureEngineer(**prep_params).fit(X, y))
                    print("!")
        
        print("!")

        # TODO: Add rank, min, max to GroupBy interactions
        # TODO: Add technique to generate few but higher-order feature interactions
        # TODO: Add technique to find binary interactions

        # EXP rounded_regression
        # if 'round_reg' in self.engineering_techniques:
        #     lgb_model_use = 'default'
        #     if len(X_num.columns) > 1 and sum(X_num.nunique() > 5)>2: # TODO: Make this a hyperparameter
        #         if f'lgb-{lgb_model_use}' not in self.scores['full']:
        #             self.scores['full'][f'lgb-{lgb_model_use}'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use)
                
        #         self.scores[f'round_reg'] = {}
        #         self.scores[f'round_reg'][f'lgb-{lgb_model_use}'], int_preds, feature_importances = self.get_lgb_performance(X, y.round(), lgb_model_type=lgb_model_use)
        #         self.scores[f'round_reg'][f'lgb-{lgb_model_use}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])
        # if best_order > 1:
        #     self.transformers.append(CatIntAdder(self.target_type, max_order=best_order, add_freq=use_freq[best_order], use_filters=False).fit(X, y))

        # self.scores['cat_groupby']['lgb-catint'], feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatGroupByAdder()], lgb_model_type='catint')

        # 1. Get all dummy-mean scores
        # irrelevant_candidates = self.dummy_mean_test(X, y)
        # self.get_dummy_mean_scores(X, y)

        # 2. Get all combination scores
        # comb_cols = self.combination_test(X, y, early_stopping=False)
        
        # 3. Get linear scores
        # linear_cols = self.interpolation_test(X, y, max_degree=1)

        # Apply categorical threshold test
        # cat_threshold_test = self.cat_threshold_test(X, y)

        # 4. Get weighted univariate scores
        # self.feature_target_rep = dict(pd.DataFrame({col: pd.DataFrame(self.scores[col]).mean().sort_values() for col in self.scores}).idxmax())
        # linear_model = Pipeline([
        #     # PowerTransformer removes a few more features to be numeric than standaradscaler, mostly the very imbalanced ones
        #     ("impute", SimpleImputer(strategy="median")),
        #     ("standardize", StandardScaler()),
        #     ("model", LinearRegression()) if self.target_type == 'regression' else ('model', LogisticRegression()),
        # ])
        # self.scores['full'] = {}
        # self.scores['full']['target_rep'] = self.cv_func(X, y, linear_model, custom_prep=[TargetRepresenter(self.feature_target_rep, self.target_type)])

        # 5. Get LGB on raw data scores
        # self.scores['full']['lgb-catint'], feature_importances = self.get_lgb_performance(X, y, lgb_model_type='catint')

        # 6. Remove entirely redundant features
        # METHOD: LGBM feature importances with very strong feature bagging fraction

        # 6. Define preprocessors to test
        # pd.DataFrame(feature_importances).mean().sort_values(ascending=False)

        ### CAT-GROUPBY INTERACTIONS
        # X_new = CatGroupByAdder().fit_transform(X, y)
        # new_cols = list(set(X_new.columns) - set(X.columns))
        # self.get_target_rep_type(X_new[new_cols], y)
        # for col in new_cols:
        #     self.feature_target_rep[col] = 'raw'
        # self.scores['cat_groupby'] = {}
        # self.scores['cat_groupby']['target_rep'] = self.cv_func(
        #     X, y, linear_model, 
        #     custom_prep=
        #     [
        #         CatGroupByAdder(),
        #         TargetRepresenter(self.feature_target_rep, self.target_type)
        #         ])
        # self.scores['cat_groupby']['lgb-catint'], feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatGroupByAdder()], lgb_model_type='catint')
        
        ### CAT INTERACTIONS
        # max_order = 2
        # X_new = CatIntAdder(max_order).fit_transform(X, y)
        # new_cols = list(set(X_new.columns) - set(X.columns))
        # self.get_target_rep_type(X_new[new_cols], y)
        # self.scores[f'cat_int{max_order}'] = {}
        # self.scores[f'cat_int{max_order}']['target_rep'] = self.cv_func(
        #     X, y, linear_model, 
        #     custom_prep=
        #     [
        #         CatIntAdder(max_order),
        #         TargetRepresenter(self.feature_target_rep, self.target_type)
        #         ])
        # self.scores[f'cat_int{max_order}']['lgb-catint'], feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(max_order)], lgb_model_type='catint')

        ### CAT-FREQUENCY DETECTION
        # X_freq = FreqAdder().fit_transform(X, y)
        # new_cols = list(set(X_freq.columns) - set(X.columns))
        # self.get_target_rep_type(X_freq[new_cols], y)
        # self.scores['cat_freq'] = {}
        # self.scores['cat_freq']['target_rep'] = self.cv_func(
        #     X, y, linear_model, 
        #     custom_prep=
        #     [
        #         FreqAdder(),
        #         TargetRepresenter(self.feature_target_rep, self.target_type)
        #         ])
        # self.scores['cat_freq']['lgb-catint'], feature_importances = self.get_lgb_performance(X, y, custom_prep=[FreqAdder()], lgb_model_type='catint')

        return self


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

    # def dummy_mean_test(self, X, y):
    #     X_use = X.copy()
    #     self.get_dummy_mean_scores(X_use, y)
    #     irrelevant_cols = []
    #     for col in X_use.columns:
    #         if self.significances[col]["test_irrelevant_mean"]>self.alpha: #'mean equal or worse than dummy'
    #             self.dtypes[col] = "irrelevant"
    #             irrelevant_cols.append(col)
        
    #     if self.verbose:
    #         print(f"\r{len(irrelevant_cols)}/{len(X_use.columns)} columns are numeric acc. to dummy_mean test.")

    #     return [col for col in X_use.columns if col not in irrelevant_cols]

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

                self.significances[col][f"combination{m_bin}_superior_to_dummy"] = self.significance_test(
                        self.scores[col][f"combination_test_{m_bin}"] - self.scores[col]["dummy"]
                    )

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

    def transform(self, X_input, mode = "overwrite", fillna_numeric=False):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)
        X = X_input.copy()
        X = X.drop(self.drop_cols, axis=1, errors='ignore')

        for transformer in self.transformers:
            X = transformer.transform(X)

        return X

if __name__ == "__main__":
    import os
    from tabprep.utils import *
    import openml
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'APS'
    for benchmark in ['TabArena']: # ["Grinsztajn", "TabArena", "TabZilla"]:
        exp_name = f"EXP_AllInOne-numint3{benchmark}"
        if False: #os.path.exists(f"{exp_name}.pkl"):
            with open(f"{exp_name}.pkl", "rb") as f:
                results = pickle.load(f)
        else:
            results = {}
            results['performance'] = {}
            results['new_cols'] = {}
            results['drop_cols'] = {}
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
            
            detector = AllInOneEngineer(        
                target_type=target_type,
                # engineering_techniques=['drop_irrelevant', 'cat_freq', 'cat_int', 'cat_groupby', 'num_int']
                engineering_techniques=['groupby'],
                use_residuals=False,
                                        )

            detector.fit(X, y)
            X_new = detector.transform(X)
            print(f"Drop columns ({len(detector.drop_cols)}): {detector.drop_cols}")
            print(f"New columns ({X_new.shape[1] - X.shape[1]}): {list(set(X_new.columns)-set(X.columns))}")
            # print(pd.DataFrame(detector.significances))
            # print(pd.DataFrame({col: pd.DataFrame(detector.scores[col]).mean().sort_values() for col in detector.scores}).transpose())
            # print(detector.linear_features.keys())
            results['performance'][data.name] = detector.scores
            results['significances'][data.name] = detector.significances
            results['new_cols'][data.name] = list(set(X_new.columns)-set(X.columns))
            results['drop_cols'][data.name] = detector.drop_cols
            
        with open(f"{exp_name}.pkl", "wb") as f:
            pickle.dump(results, f)
        break