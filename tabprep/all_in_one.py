import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, log_loss
from tabprep.proxy_models import TargetMeanClassifier, TargetMeanRegressor, UnivariateLinearRegressor, UnivariateLogisticClassifier, PolynomialLogisticClassifier, PolynomialRegressor, UnivariateThresholdClassifier, MultiFeatureTargetMeanClassifier, MultiFeatureUnivariateLogisticClassifier, LightGBMBinner, KMeansBinner, TargetMeanRegressorCut, TargetMeanClassifierCut, CustomLinearModel
from tabprep.utils import p_value_wilcoxon_greater_than_zero, clean_feature_names, clean_series, make_cv_function, p_value_sign_test_median_greater_than_zero, p_value_ttest_greater_than_zero
import time
from category_encoders import LeaveOneOutEncoder
from tabprep.base_preprocessor import BasePreprocessor

from tabprep.preprocessors import TargetRepresenter, FreqAdder, CatIntAdder, CatGroupByAdder, SVDConcatTransformer, CatAsNumTransformer, CatOHETransformer, CatLOOTransformer, DuplicateCountAdder, DuplicateContentLOOEncoder, LinearFeatureAdder, OOFLinearFeatureAdder
from tabprep.num_interaction import NumericalInteractionDetector
from tabprep.groupby_interactions import GroupByFeatureEngineer
from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator

class AllInOneEngineer(BasePreprocessor):
    def __init__(self, target_type, use_residuals=False,
                 n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                 combination_test_min_bins=2, combination_test_max_bins=2048, binning_strategy='lgb',                 
                 engineering_techniques: list=[],
                 lgb_model_type='default',
                 ):
        self.original_target_type = target_type
        # NOTE: Idea: Reinvent interaction test by first categorizing a dataset into 1) there are already clear categorical features; 2) there are only numerical features; 3) there are only numerical features, but many categorical candidates. And based on that use different interaction tests.
        # TODO: Make sure that everything is deterministic given a fixed seed
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose, multi_as_bin=False)
        self.use_residuals = use_residuals
        self.combination_test_min_bins = combination_test_min_bins
        self.combination_test_max_bins = combination_test_max_bins
        self.binning_strategy = binning_strategy
        self.engineering_techniques = engineering_techniques
        self.lgb_model_type = lgb_model_type


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
        self.base_transformers = []
        self.transformers_unfitted = []
        self.drop_cols = []
        self.post_predict_duplicate_mapping = False
        
        # self.target_transformers = []
        self.linear_residuals = None
        self.dupe_map = {}
        self.map_duplicates = False

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

    def cat_as_num(self, X_in):
        X_out = X_in.copy()
        for col in X_out.select_dtypes(include=['object', 'category']).columns:
            num_convertible = pd.to_numeric(X_out[col].dropna(), errors='coerce').notna().all()
            if num_convertible:
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce')
            else:
                X_out[col] = OrdinalEncoder().fit_transform(X_out[[col]]).flatten()
        return X_out

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

        # 0. Apply AG preprocessing and get the base scores
        old_cols = X.columns.tolist()
        X = DropDuplicatesFeatureGenerator().fit_transform(X)
        if X.shape[1] < len(old_cols):
            self.drop_cols.extend([col for col in old_cols if col not in X.columns])
        X = X.drop(columns=self.drop_cols, errors='ignore', axis=1)

        # FIXME: Need a global function to filter to engineering techniques that will be applied and only fit the full model if necessary
        self.scores['full'] = {}
        # if sum(X.nunique()==2)!=X.shape[1]:
        self.scores['full'][f'lgb-{self.lgb_model_type}'], base_preds, base_feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type)

        if 'quantile_reg' in self.engineering_techniques:
            self.scores['full'][f'quantile_1'], q_preds, q_imp = self.get_lgb_performance(X, y, lgb_model_type='quantile_1')
            self.scores['full'][f'quantile_3'], q_preds, q_imp = self.get_lgb_performance(X, y, lgb_model_type='quantile_3')
            self.scores['full'][f'quantile_5'], q_preds, q_imp = self.get_lgb_performance(X, y, lgb_model_type='quantile_5')
            self.scores['full'][f'quantile_9'], q_preds, q_imp = self.get_lgb_performance(X, y, lgb_model_type='quantile_9')
            # if np.mean(self.scores['full'][f'quantile_9']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
            #     pass

        if 'linear_residuals' in self.engineering_techniques:
            if self.target_type != 'multiclass':
                self.scores['full'][f'linear'], linear_preds = self.cv_func(X, y, Pipeline([
                    ('model', CustomLinearModel(target_type=self.target_type))
                ]), return_preds=True)
                if np.mean(self.scores['full'][f'linear']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                    all_linear_preds = pd.concat(linear_preds).sort_index()
                    lin_residuals = y - all_linear_preds

                    self.scores['lin_residuals'] = {}
                    self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'], res_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, residuals='linear_residuals', scale_y='linear_residuals')
                    # self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'], res_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, residuals=lin_residuals)
                    # if target_type != 'regression':
                    #     self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[res_pred.index],lin_pred+res_pred) for lin_pred,res_pred in zip(linear_preds,res_preds)])                
                    if np.mean(self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                        self.linear_residual_model = CustomLinearModel(target_type=self.target_type).fit(X, y)
                        # self.linear_residual_model = CustomLinearModel(target_type=self.target_type)
                        self.linear_residuals = True

                        print(pd.DataFrame({col: pd.DataFrame(self.scores[col]).mean().sort_values() for col in self.scores}).transpose().sort_values('lgb-default'))

        if 'duplicate_mapping' in self.engineering_techniques and X.shape[1] < 1000:
            n_X_duplicates = X.duplicated().mean()
            n_Xy_duplicates = pd.concat([X,y],axis=1).duplicated().mean()

            # if n_X_duplicates > 0.0 and self.target_type != 'multiclass': #and n_X_duplicates == n_Xy_duplicates:
            if n_X_duplicates > 0.1 and (n_X_duplicates-n_Xy_duplicates)<0.01 and self.target_type != 'multiclass': #and n_X_duplicates == n_Xy_duplicates:
                self.post_predict_duplicate_mapping = True
                X_str = X.astype(str).sum(axis=1)
                if self.target_type in ['regression', 'binary']:
                    self.dupe_map = dict(y.groupby(X_str).mean())
                    if round(n_X_duplicates,2) != round(n_Xy_duplicates,2):
                        cnt_map = dict(y.groupby(X_str).count())
                        self.dupe_map = {k: v for k, v in self.dupe_map.items() if cnt_map[k] > 1}
                        dupe_min = dict(y.groupby(X_str).min())
                        dupe_max = dict(y.groupby(X_str).max())
                        self.dupe_map = {k: v for k, v in self.dupe_map.items() if dupe_min[k]==dupe_max[k]}
                # if self.target_type == 'binary':
                #     self.dupe_map = dict(y.groupby(X_str).mean())
                #     dupe_min = dict(y.groupby(X_str).min())
                #     dupe_max = dict(y.groupby(X_str).max())
                #     self.dupe_map = {k: v for k, v in self.dupe_map.items() if dupe_min[k]==dupe_max[k]}
                elif self.target_type == 'multiclass':
                    # TODO: Implement approach properly for multiclass
                    self.post_predict_duplicate_mapping = False
                    y_orig = pd.Series(self.multiclass_encoder.inverse_transform(y))
                    self.dupe_map = dict(y_orig.groupby(X_str).apply(lambda x: (sum(x==y_orig.loc[x.idxmax()])/len(x))))
                    self.dupe_mode = dict(y_orig.groupby(X_str).apply(lambda x: x.mode().values[0]))

                if len(self.dupe_map)>0:
                    print(n_X_duplicates, n_Xy_duplicates, len(self.dupe_map))
                # TODO: Add the option to use post_predict transformers in my CV
                # TODO: Add post_predict transformers to the pipeline
                # if f'lgb-{lgb_model_use}' not in self.scores['full']:
                #     self.scores['full'][f'lgb-{lgb_model_use}'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use)

        # 1. Detect irrelevant features
        # TODO: Add AG method here
        if 'drop_irrelevant' in self.engineering_techniques:
            self.irrelevant_feature_importances = pd.DataFrame(base_feature_importances).mean().sort_values()

            # TODO: Add as a preprocessor
            self.drop_cols.extend(self.irrelevant_feature_importances[self.irrelevant_feature_importances < self.alpha].index.tolist())
            self.base_transformers.append('drop_irrelevant')

        X = X.drop(columns=self.drop_cols, errors='ignore', axis=1)
        X_bin = X[X.columns[X.nunique()<=2]]
        X_cat = X[X.columns[X.nunique()>2]].select_dtypes(include=['object', 'category'])
        X_num = X[X.columns[X.nunique()>2]].select_dtypes(exclude=['object', 'category'])

        assert X_cat.shape[1] + X_num.shape[1] + X_bin.shape[1] ==  X.shape[1], "Not all features covered through num/cat selection."

        # Cat-as-num detection
        if 'cat_as_num' in self.engineering_techniques:            
            if len(X_cat.columns) > 0:
                self.scores['cat_as_num'] = {}
                self.scores['cat_as_num'][f'lgb-{self.lgb_model_type}'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, custom_prep=[CatAsNumTransformer()])

                if np.mean(self.scores['cat_as_num'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                    self.base_transformers.append(CatAsNumTransformer().fit(X))
                    X = self.cat_as_num(X)
                    X_bin = X[X.columns[X.nunique()<=2]]
                    X_cat = X[X.columns[X.nunique()>2]].select_dtypes(include=['object', 'category'])
                    X_num = X[X.columns[X.nunique()>2]].select_dtypes(exclude=['object', 'category'])

        # Cat-as-loo detection
        # if 'cat_as_loo' in self.engineering_techniques:
        #     if len(X_cat.columns) > 0:
        #         self.scores['cat_as_loo'] = {}
        #         self.scores['cat_as_loo']['lgb-default'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, custom_prep=[CatLOOTransformer()])

        #         if np.mean(self.scores['cat_as_loo']['lgb-default']) > np.mean(self.scores['full']['lgb-default']):
        #             self.base_transformers.append(CatLOOTransformer().fit(X))

        # 2. Categorical frequency
        if 'cat_freq' in self.engineering_techniques:
            if len(X_cat.columns) > 0:
                cat_freq = FreqAdder()
                candidate_cols = cat_freq.filter_candidates_by_distinctiveness(X_cat)
                if len(candidate_cols) > 0:
                    self.scores['cat_freq'] = {}
                    self.scores['cat_freq'][f'lgb-{self.lgb_model_type}'], preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[FreqAdder(candidate_cols=candidate_cols)], lgb_model_type=self.lgb_model_type)

                    if np.mean(self.scores['cat_freq'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                        self.transformers.append(FreqAdder(candidate_cols=candidate_cols).fit(X, y))
                        self.transformers_unfitted.append(FreqAdder(candidate_cols=candidate_cols))
        ######## EXPERIMENTAL PART ########
        if 'SVD' in self.engineering_techniques:
            self.scores[f'svd'] = {}
            self.scores[f'svd'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[SVDConcatTransformer()], lgb_model_type=self.lgb_model_type)

            if np.mean(self.scores[f'svd'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                self.transformers.append(SVDConcatTransformer().fit(X, y))
                self.transformers_unfitted.append(SVDConcatTransformer())

        # elif 'SVD-TTA' in self.engineering_techniques:
        #     self.transformers.append('SVD-TTA')
        # self.scores['full']['lgb-default-numeric'], preds, feature_importances = self.get_lgb_performance(X.astype(float), y, lgb_model_type='default')        
        # self.scores['full']['lgb-default-tabpfn-SVD'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type='default', custom_prep=[SVDConcatTransformer()])
        # self.scores['full']['lgb-default-tabpfn-SVD-TTA'], preds, feature_importances = self.get_lgb_performance(SVDConcatTransformer().fit_transform(X.astype(float)), y, lgb_model_type='default', )
        # self.scores['full']['lgb-default-tabpfn-SVD'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type='default', custom_prep=[SVDConcatTransformer()])
        # self.scores['full']['lgb-default-tabpfn-SVD-TTA'], preds, feature_importances = self.get_lgb_performance(SVDConcatTransformer().fit_transform(X), y, lgb_model_type='default', )

        # # 3. Categorical interactions
        if 'cat_int' in self.engineering_techniques:
            if len(X_cat.columns) > 1 and sum(X_cat.nunique() > 5)>2: # TODO: Make this a hyperparameter
                if self.use_residuals:
                    residuals = y-pd.concat(base_preds).sort_index()
                curr_comp = 'full'
                best_catint_order = 1
                use_freq = {}
                for order in range(2,4): # TODO: Add max_order as parameter
                    if len(X_cat.columns) >= order:
                        self.scores[f'cat_int{order}'] = {}
                        self.scores[f'cat_int{order}_andFreq'] = {}
                        # self.scores[f'cat_int{order}_onlyFreq'] = {}

                        if self.use_residuals:
                            self.scores[f'cat_int{order}'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, use_filters=False)], lgb_model_type=self.lgb_model_type, residuals=residuals)
                            if target_type != 'regression':
                                self.scores[f'cat_int{order}'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])
                            # TODO: Add some additional test to not always attempt +freq
                            # I.e., only if the previous order was better with freq
                            self.scores[f'cat_int{order}_andFreq'][f'lgb-{self.lgb_model_type}'], intfreq_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, add_freq=True, use_filters=False)], lgb_model_type=self.lgb_model_type, residuals=residuals)
                            if target_type != 'regression':
                                self.scores[f'cat_int{order}_andFreq'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in intfreq_preds])
                        else:
                            self.scores[f'cat_int{order}'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, use_filters=False)], lgb_model_type=self.lgb_model_type)
                            self.scores[f'cat_int{order}_andFreq'][f'lgb-{self.lgb_model_type}'], intfreq_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, add_freq=True, use_filters=False)], lgb_model_type=self.lgb_model_type)

                        andFreq_beats_int = np.mean(self.scores[f'cat_int{order}_andFreq'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores[f'cat_int{order}'][f'lgb-{self.lgb_model_type}'])
                        andFreq_beats_base = np.mean(self.scores[f'cat_int{order}_andFreq'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores[curr_comp][f'lgb-{self.lgb_model_type}'])
                        int_beats_base = np.mean(self.scores[f'cat_int{order}'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores[curr_comp][f'lgb-{self.lgb_model_type}'])

                        if andFreq_beats_int and andFreq_beats_base:
                            curr_comp = f'cat_int{order}_andFreq'
                            best_catint_order = order
                            use_freq[order] = True
                            residuals = y - pd.concat(intfreq_preds).sort_index()
                        elif int_beats_base:
                            curr_comp = f'cat_int{order}'
                            best_catint_order = order
                            use_freq[order] = False
                            residuals = y - pd.concat(int_preds).sort_index()
                        else:
                            break
                if best_catint_order > 1:
                    # TODO: Add post-hoc adjustment to drop unused features
                    self.transformers.append(CatIntAdder(self.target_type, max_order=best_catint_order, add_freq=use_freq[best_catint_order], use_filters=False).fit(X, y))
                    self.transformers_unfitted.append(CatIntAdder(self.target_type, max_order=best_catint_order, add_freq=use_freq[best_catint_order], use_filters=False))

        # 4. Categorical GroupBy interactions
        if 'cat_groupby' in self.engineering_techniques:
            min_cardinality = 5 # TODO: Make this a hyperparameter
            if len(X_cat.columns) > 1 and sum(X_cat.nunique() >= min_cardinality)>2: # TODO: Test to filter low-cardinality for groupby
                if self.use_residuals:
                    residuals = y-pd.concat(base_preds).sort_index()
                
                self.scores[f'cat_groupby'] = {}
                if self.use_residuals:
                    self.scores[f'cat_groupby'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatGroupByAdder(min_cardinality=min_cardinality)], lgb_model_type=self.lgb_model_type, residuals=residuals)
                    if target_type != 'regression':
                        self.scores[f'cat_groupby'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])                    
                else:
                    self.scores[f'cat_groupby'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[CatGroupByAdder(min_cardinality=min_cardinality)], lgb_model_type=self.lgb_model_type)

                if np.mean(self.scores[f'cat_groupby'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                    self.transformers.append(CatGroupByAdder(min_cardinality=min_cardinality).fit(X, y))

        # 5. Numerical interactions
        num_int=False
        if 'num_int' in self.engineering_techniques:
            if len(X_num.columns) > 1 and sum(X_num.nunique() > 5)>2: # TODO: Make this a hyperparameter
                if self.use_residuals:
                    residuals = y-pd.concat(base_preds).sort_index()
                curr_comp = 'full'
                
                imp_df = pd.DataFrame(base_feature_importances).mean()
                candidate_cols = imp_df.loc[X_num.columns].sort_values(ascending=False).head(100).index.tolist()
                
                prep_args = {}
                best_order = 1
                for order in range(2,3): # TODO: Add max_order as parameter
                    if len(X_num.columns) >= order:
                        self.scores[f'num_int{order}'] = {}
                        
                        prep_args[order] = {
                            'target_type': self.target_type, 
                            'max_order': 3, 
                            'num_operations': 'all',
                            'use_mvp': False,
                            'corr_thresh': .95,
                            'select_n_candidates': 2000,
                            'apply_filters': False,
                            'candidate_cols': candidate_cols,
                            'min_cardinality': 3,
                        }

                        preprocessor = NumericalInteractionDetector(
                            **prep_args[order]
                        )

                        if self.use_residuals:
                            self.scores[f'num_int{order}'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=self.lgb_model_type, residuals=residuals)
                            if target_type != 'regression':
                                self.scores[f'num_int{order}'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])                    
                        else:
                            self.scores[f'num_int{order}'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=self.lgb_model_type)

                        if np.mean(self.scores[f'num_int{order}'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores[curr_comp][f'lgb-{self.lgb_model_type}']):
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
                    num_int = True
                
        
        # 6. Cat-by-NUM GroupBy interactions
        if 'groupby' in self.engineering_techniques:
            if len(X_cat.columns) > 0 and any(X_cat.nunique() > 5) and len(X_num.columns) > 0 and any(X_num.nunique() > 5): # TODO: Test to filter low-cardinality for groupby
                if self.use_residuals:
                    residuals = y-pd.concat(base_preds).sort_index()
                
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
                    self.scores[f'groupby'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=self.lgb_model_type, residuals=residuals)
                    if target_type != 'regression':
                        self.scores[f'groupby'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])                    
                else:
                    self.scores[f'groupby'][f'lgb-{self.lgb_model_type}'], int_preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=self.lgb_model_type)

                if np.mean(self.scores[f'groupby'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                    self.transformers.append(GroupByFeatureEngineer(**prep_params).fit(X, y))
                    print("!")        

        # 7. Binary FE
        if 'bin_summarize' in self.engineering_techniques:
            if self.use_residuals:
                residuals = y-pd.concat(base_preds).sort_index()
            imp_df = pd.DataFrame(base_feature_importances).mean().sort_values(ascending=False)
            
            X_new = X.copy()    
            for i in range(1, 10):
                start_col = imp_df.index[i]
                X_curr = X[start_col].astype(float)
                curr_ll = log_loss(y,LeaveOneOutEncoder().fit_transform(X[start_col],y))
                use_cols = [start_col]
                drop_cols = imp_df.index[:i].tolist()
                new_ll = -np.inf
                for col in imp_df.index[i:]:
                    print(f"\rBinary FE: {len(use_cols)}/{imp_df.index[i:]} columns processed, current performance={curr_ll:.4f}, new performance={new_ll:.4f}", end="", flush=True)
                    X_cand = X.drop(columns=drop_cols+use_cols, errors='ignore').astype(float)
                    loo = LeaveOneOutEncoder().fit_transform((X_cand.transpose() + X_curr.values).transpose().astype('U'), y)
                    loo_perf = pd.Series({col: log_loss(y,loo[col]) for col in loo.columns}).sort_values()

                    new_ll = loo_perf.iloc[0]
                    if new_ll < curr_ll:
                        use_cols.append(loo_perf.index[0])
                        X_curr = X_curr + X_cand[loo_perf.index[0]]
                        curr_ll = new_ll
                    else:
                        X_new[f'sum{i}'] = X[use_cols].astype(float).sum(axis=1)
                        break

            self.scores['bin-sum'] = {}
            self.scores['bin-sum'][f'lgb-{self.lgb_model_type}'], full_preds, feature_importances = self.get_lgb_performance(X_new, y, lgb_model_type=self.lgb_model_type)

        # if 'dupe_detection' in self.engineering_techniques:            
        #     lgb_model_use = 'default'
        #     if X.duplicated().sum() > 0:
        #         self.scores['dupe_detect'] = {}
        #         if num_int:
        #             self.scores['dupe_detect'][f'lgb-{self.lgb_model_type}'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use, custom_prep=[DuplicateCountAdder(), NumericalInteractionDetector(**prep_args[best_order])])
        #             reference_config = f'num_int2'
        #         else:
        #             self.scores['dupe_detect'][f'lgb-{self.lgb_model_type}'], preds, feature_importances = self.get_lgb_performance(X, y, custom_prep=[DuplicateCountAdder()], lgb_model_type=lgb_model_use)
        #             reference_config = 'full'
        #         if np.mean(self.scores['dupe_detect'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores[reference_config][f'lgb-{self.lgb_model_type}']):
        #             self.base_transformers.append(DuplicateCountAdder().fit(X))

        # if 'linear_residuals' in self.engineering_techniques:
        #     if self.target_type != 'multiclass':
        #         self.scores['full'][f'linear'], linear_preds = self.cv_func(X, y, Pipeline([
        #             ('model', CustomLinearModel(target_type=self.target_type))
        #         ]), return_preds=True)
        #         if np.mean(self.scores['full'][f'linear']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
        #             all_linear_preds = pd.concat(linear_preds).sort_index()
        #             lin_residuals = y - all_linear_preds

        #             self.scores['lin_residuals'] = {}
        #             self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'], res_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, residuals='linear_residuals', scale_y='linear_residuals')
        #             # self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'], res_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, residuals=lin_residuals)
        #             # if target_type != 'regression':
        #             #     self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[res_pred.index],lin_pred+res_pred) for lin_pred,res_pred in zip(linear_preds,res_preds)])                
        #             if np.mean(self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
        #                 # self.linear_residual_model = CustomLinearModel(target_type=self.target_type).fit(X, y)
        #                 self.linear_residual_model = CustomLinearModel(target_type=self.target_type)
        #                 self.linear_residuals = True

        #                 print(pd.DataFrame({col: pd.DataFrame(self.scores[col]).mean().sort_values() for col in self.scores}).transpose().sort_values('lgb-default'))
                
                # if self.target_type == 'regression' and 'linear_feature' in self.engineering_techniques:


        if 'linear_feature' in self.engineering_techniques:
            get_lm = lambda: LinearFeatureAdder(self.target_type, linear_model_type='lasso')
            self.scores['full'][f'linear_feature'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, custom_prep=[get_lm()])
            # self.scores['full'][f'linear_feature'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use, custom_prep=[LinearFeatureAdder(self.target_type, linear_model_type='lasso')])

            if np.mean(self.scores[f'full'][f'linear_feature']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                self.transformers.append(get_lm().fit(X, y))
                print("!")

        # if self.target_type == 'regression':
        #     self.scores['full'][f'closest'], close_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use, reg_assign_closest_y=True)

        if self.target_type == 'regression' and 'log_target' in self.engineering_techniques:
            lgb_model_use = 'default'
            if f'lgb-{lgb_model_use}' not in self.scores['full']:
                self.scores['full'][f'lgb-{lgb_model_use}'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use)
            self.scores['full'][f'log_target'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use, scale_y='log')

            # if np.mean(self.scores[f'full'][f'log_target']) > np.mean(self.scores['full'][f'lgb-{lgb_model_use}']):
            #     self.transformers.append(LinearFeatureAdder(self.target_type).fit(X, y))
            #     print("!")            


        # Cat-as-ohe detection
        if 'freq_as_ohe' in self.engineering_techniques:            
            if len(X_cat.columns) > 0: # and X_cat.nunique().max() > 100:
                from tabprep.preprocessors import FrequencyOHE
                self.scores['freq_as_ohe'] = {}
                self.scores['freq_as_ohe'][f'lgb-{self.lgb_model_type}'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, custom_prep=[FrequencyOHE(min_freq=20)])

                if np.mean(self.scores['freq_as_ohe'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                    print('Use OHE')
                    self.transformers.append(CatOHETransformer().fit(X))

        # Cat-as-ohe detection
        if 'cat_as_ohe' in self.engineering_techniques:            
            if len(X_cat.columns) > 0: # and X_cat.nunique().max() > 100:
                self.scores['cat_as_ohe'] = {}
                # if 'cat_int' in self.engineering_techniques and best_catint_order > 1:
                #     self.scores['cat_as_ohe'][f'lgb-{self.lgb_model_type}'], preds, feature_importances = self.get_lgb_performance(
                #         X, y, lgb_model_type=self.lgb_model_type, 
                #         custom_prep=[CatIntAdder(self.target_type, max_order=best_catint_order, add_freq=use_freq[best_catint_order], use_filters=False), 
                #                      CatOHETransformer()])
                # else:
                self.scores['cat_as_ohe'][f'lgb-{self.lgb_model_type}'], preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, custom_prep=[CatOHETransformer()])

                if np.mean(self.scores['cat_as_ohe'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                    print('Use OHE')
                    self.transformers.append(CatOHETransformer().fit(X))


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
    

    def transform(self, X_input, mode = "overwrite", fillna_numeric=False):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)
        X = X_input.copy()
        X = X.drop(self.drop_cols, axis=1, errors='ignore')

        for transformer in self.base_transformers:
            X = transformer.transform(X)

        X_new = X.copy()
        for transformer in self.transformers:
            X_feat = transformer.transform(X)

            new_cols = list(set(X_feat.columns) - set(X.columns))
            X_new = pd.concat([X_new, X_feat[new_cols]], axis=1)

        return X_new
    
    # def transform_y(self, y_input):
    #     y = y_input.copy()
    #     if self.linear_residuals:
    #         y_new = self.target_transformers[0].transform(y)

    #     y_new = pd.Series(y_new, index=y.index, name=y.name)

    #     return y_new


if __name__ == "__main__":
    import os
    from tabprep.utils import *
    import openml
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'physio'
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
            results['dupe_map'] = {}

        tids, dids = get_benchmark_dataIDs(benchmark)  

        remaining_cols = {}

        for tid, did in zip(tids, dids):
            task = openml.tasks.get_task(tid)  # to check if the datasets are available
            data = openml.datasets.get_dataset(did)  # to check if the datasets are available
            if dataset_name not in data.name:
                continue
            
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
            
            detector = AllInOneEngineer(        
                target_type=target_type,
                # engineering_techniques=['cat_as_num', 'cat_freq', 'cat_int', 'cat_groupby', 'num_int', 'groupby', 'linear_residuals'],
                engineering_techniques=['quantile_reg'],
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
            results['dupe_map'][data.name] = len(detector.dupe_map)
        with open(f"{exp_name}.pkl", "wb") as f:
            pickle.dump(results, f)
        break