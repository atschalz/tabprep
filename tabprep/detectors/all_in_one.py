import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss

from tabprep.proxy_models import  CustomLinearModel
from tabprep.utils.modeling_utils import make_cv_function, clean_feature_names
from tabprep.detectors.base_preprocessor import BasePreprocessor
from tabprep.detectors.num_interaction import NumericalInteractionDetector
from tabprep.detectors.groupby_interactions import GroupByFeatureEngineer

from category_encoders import LeaveOneOutEncoder
from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator

from tabprep.utils.modeling_utils import adapt_lgb_params, adjust_target_format
from tabprep.preprocessors.misc import TargetRepresenter
from tabprep.preprocessors.frequency import FrequencyEncoder
from tabprep.preprocessors.categorical import CatIntAdder, CatGroupByAdder, OneHotPreprocessor, CatLOOTransformer
from tabprep.preprocessors.type_change import CatAsNumTransformer
from tabprep.preprocessors.multivariate import SVDPreprocessor, DuplicateCountAdder, LinearFeatureAdder

# from tabprep.preprocessors.misc 

from typing import Union, Optional, List, Callable, Literal, Dict, Any

# TODO: Define hyperparameters for the techniques and consider more general format per technique
RegisteredTechniques: Dict[str, Dict[str, Any]] = {
    'drop_irrelevant': {'irrelevant_threshold': 0.01}, # Filter techniques that are not used in any fold
    'cat_freq': {},
    'cat_as_num': {},
    'cat_as_loo': {},
    'cat_as_ohe': {},
    'cat_int': {},
    'cat_groupby': {},
    'num_int': {},
    'groupby': {},
    'duplicate_mapping': {},
    'linear_residuals': {},
    'quantile_reg': {},
    'SVD': {}
}

# TODO: Define a better name once overall package structure is clear
class AllInOneEngineer():
    def __init__(
            self, 
            target_type: Literal["binary", "multiclass", "regression"], 
            engineering_techniques: Optional[List[str]] = None,
            use_residuals: bool = False,
            n_folds: int = 5,
            lgb_model_type: Literal["default"] = "default", # TODO: Add supported ones; TODO: Add option to use own hyperparameters
            min_cardinality: int = 6,
            early_stopping_rounds: int = 20,
            technique_params: Optional[Dict[str, Dict[str, Any]]] = RegisteredTechniques,
            verbose: bool = False,
            **kwargs
            ):
        self.target_type = target_type
        self.registered_techniques = list(RegisteredTechniques.keys())
        self.use_residuals = use_residuals
        self.lgb_model_type = lgb_model_type
        self.min_cardinality = min_cardinality
        self.technique_params = technique_params
        self.verbose = verbose

        if engineering_techniques is None:
            self.engineering_techniques = []
            import warnings
            warnings.warn("No engineering techniques provided. No feature engineering will be performed.")
        else:
            self.engineering_techniques = engineering_techniques

        # Define evaluation functions
        self.cv_func = make_cv_function(target_type=self.target_type, early_stopping_rounds=early_stopping_rounds, n_folds=n_folds, verbose=verbose)
        if self.target_type=='binary':
            self.scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
        elif self.target_type=='regression':
            self.scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
        elif self.target_type=='multiclass':
            self.scorer = lambda ytr, ypr: -log_loss(ytr, ypr) # r2_score

        # Predefine output variables
        self.preprocessors_running = [] # TODO: Add category dtype assignment, and potentially also others
        self.scores = {} # {data_version: model_type: [scores across folds]}
        self.drop_cols = []
        self.transformers = [] # Transformers that just add features (e.g., CatInt)
        self.base_transformers = [] # Transformers that modify the original features (e.g., CatAsNum) or are applied in sequential order

        # TODO: These parameters are redundant, remove once logic in tabrepo is finalized
        self.post_predict_duplicate_mapping = False
        
        self.dupe_map = {}

    def get_lgb_performance(
            self, 
            X_cand_in: pd.DataFrame, 
            y_in: pd.Series, 
            lgb_model_type: str = None, 
            custom_prep: List[Any] = None, 
            residuals: str | pd.Series = None, 
            scale_y: str = None,
            reg_assign_closest_y: bool = False,
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

    def prefilter_X(self, X: pd.DataFrame):
        # TODO: Remove constant columns
        old_cols = X.columns.tolist()
        X = DropDuplicatesFeatureGenerator().fit_transform(X)
        if X.shape[1] < len(old_cols):
            self.drop_cols.extend([col for col in old_cols if col not in X.columns])
        X = X.drop(columns=self.drop_cols, errors='ignore', axis=1)
        return X

    def _get_column_types(self, X: pd.DataFrame):
        """
        Utility method to get binary, categorical, and numerical columns based on cardinality and dtype.
        """
        bin_cols = X.columns[X.nunique() <= 2].tolist()
        cat_cols = X.columns[(X.nunique() > 2) & (X.dtypes.isin(['object', 'category']))].tolist()
        num_cols = X.columns[(X.nunique() > 2) & (~X.dtypes.isin(['object', 'category']))].tolist()
        assert len(cat_cols) + len(num_cols) + len(bin_cols) ==  X.shape[1], "Not all features covered through num/cat selection."
        return bin_cols, cat_cols, num_cols

    def filter_techniques(self, X: pd.DataFrame, y: pd.Series):
        # Use utility method to get column types
        bin_cols, cat_cols, num_cols = self._get_column_types(X)
        applicable_techniques = []
        for t in self.engineering_techniques:
            if t not in self.registered_techniques:
                raise ValueError(f"Engineering technique {t} not recognized. Available techniques: {self.registered_techniques}")
            
            # Always applicable
            if t in ['drop_irrelevant']:
                applicable_techniques.append(t)
            
            # Not applicable to classification
            if t in ['quantile_reg'] and self.target_type in ['binary', 'multiclass']:
                continue
            # Not applicable to multiclass
            if t in ['linear_residuals', 'duplicate_mapping'] and self.target_type in ['multiclass']:
                continue

            if t == 'duplicate_mapping':
                if X.shape[1] >= 1000: # It's unlikely that there are duplicates in very wide datasets
                    continue

                n_X_duplicates = X.duplicated().mean()
                n_Xy_duplicates = pd.concat([X,y],axis=1).duplicated().mean()
                # Only apply to datasets with more than 10% duplicates and where almost all duplicates have the same target value
                # TODO: Rethink duplicate detection strategy
                if n_X_duplicates <= 0.1 or (n_X_duplicates-n_Xy_duplicates)>=0.01: 
                    continue

            if t == 'duplicate_count':
                n_X_duplicates = X.duplicated().mean()

                if n_X_duplicates<=0.001:
                    continue

            # Only applicable if categorical features are present
            if t in ['cat_as_num', 'cat_as_loo', 'cat_as_ohe', 'cat_freq', 'cat_int', 'cat_groupby']:
                if len(cat_cols) == 0:
                    continue

                if t == 'cat_freq':
                    # Only test extensively if there are candidate categorical features
                    cat_freq = FrequencyEncoder()
                    candidate_cols = cat_freq.filter_candidates_by_distinctiveness(X[cat_cols])
                    if len(candidate_cols) == 0:
                        continue

                # TODO: Reconsider filtering by cardinality
                if t == 'cat_int' and (len(cat_cols) < 2 or (X[cat_cols].nunique() > 5).sum() < 2):
                    continue

                if t == 'cat_groupby':
                    if not (X[cat_cols].nunique() >= self.min_cardinality).sum() > 2:
                        continue
            
            if t in ['num_int']:
                if len(num_cols) < 2: # TODO: Test impact of cardinality filter or sum(X[num_cols].nunique() > 5)<2:
                    continue
            
            if t in ['groupby']:
                if len(cat_cols) == 0 or len(num_cols) == 0:
                    continue
                if not (X[cat_cols].nunique() >= self.min_cardinality).any() or not (X[num_cols].nunique() >= self.min_cardinality).any():
                    continue

            if t == 'bin_summarize' and len(bin_cols) < 20: 
                continue

            applicable_techniques.append(t)

        return applicable_techniques

    def adjust_detection_order(self, techniques: List[str]):
        # Ensure that duplicate_count is always first or second if drop_irrelevant is also used
        if 'duplicate_count' in techniques and techniques[0] != 'duplicate_count':
            techniques.remove('duplicate_count')
            techniques = ['duplicate_count'] + techniques
        
        # Ensure that drop_irrelevant is always first
        if 'drop_irrelevant' in techniques and techniques[0] != 'drop_irrelevant':
            techniques.remove('drop_irrelevant')
            techniques = ['drop_irrelevant'] + techniques

        # # Ensure that linear_residuals is always last
        # if 'linear_residuals' in techniques:
        #     techniques.remove('linear_residuals')
        #     techniques = techniques + ['linear_residuals']

        return techniques

    def get_base_scores(self, X: pd.DataFrame, y: pd.Series):
        if self.verbose:
            print('Fit base model.')
        self.scores['full'] = {}
        # TODO: Check whether we need to return preds & importances
        # TODO: Adjust names of self.scores entries
        res = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type)
        self.scores['full'][f'lgb-{self.lgb_model_type}'] = res['scores']
        base_preds, base_feature_importances = res['preds'], res['importances']
        return base_feature_importances, base_preds

    def detect_quantile_regression(self, X: pd.DataFrame, y: pd.Series):
        # TODO: Implement quantile regression properly
        self.scores['full'][f'quantile_1'] = self.get_lgb_performance(X, y, lgb_model_type='quantile_1')['scores']
        self.scores['full'][f'quantile_3'] = self.get_lgb_performance(X, y, lgb_model_type='quantile_3')['scores']
        self.scores['full'][f'quantile_5'] = self.get_lgb_performance(X, y, lgb_model_type='quantile_5')['scores']
        self.scores['full'][f'quantile_9'] = self.get_lgb_performance(X, y, lgb_model_type='quantile_9')['scores']

    def detect_linear_residuals(self, X: pd.DataFrame, y: pd.Series):
        res = self.cv_func(X, y, Pipeline([
            ('model', CustomLinearModel(target_type=self.target_type))
        ]), return_preds=True)
        self.scores['full'][f'linear'], linear_preds = res['scores'], res['preds']
        
        if True: #np.mean(self.scores['full'][f'linear']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
            all_linear_preds = pd.concat(linear_preds).sort_index()
            lin_residuals = y - all_linear_preds

            self.scores['lin_residuals'] = {}
            self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'] = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, residuals='linear_residuals', scale_y='linear_residuals')['scores']
            # self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'], res_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, residuals=lin_residuals)
            # if target_type != 'regression':
            #     self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[res_pred.index],lin_pred+res_pred) for lin_pred,res_pred in zip(linear_preds,res_preds)])                
            if np.mean(self.scores[f'lin_residuals'][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                self.linear_residual_model = CustomLinearModel(target_type=self.target_type).fit(X, y)
                # self.linear_residual_model = CustomLinearModel(target_type=self.target_type)
                

                # print(pd.DataFrame({col: pd.DataFrame(self.scores[col]).mean().sort_values() for col in self.scores}).transpose().sort_values('lgb-default'))

    def detect_duplicate_mapping(self, X: pd.DataFrame, y: pd.Series):
        n_X_duplicates = X.duplicated().mean()
        n_Xy_duplicates = pd.concat([X,y],axis=1).duplicated().mean()

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

        # if len(self.dupe_map)>0:
        #     print(n_X_duplicates, n_Xy_duplicates, len(self.dupe_map))
        # TODO: Add the option to use post_predict transformers in my CV
        # TODO: Add post_predict transformers to the pipeline
        # if f'lgb-{lgb_model_use}' not in self.scores['full']:
        #     self.scores['full'][f'lgb-{lgb_model_use}'], full_preds, feature_importances = self.get_lgb_performance(X, y, lgb_model_type=lgb_model_use)
        # 

    def detect_cat_interactions(self, X: pd.DataFrame, y: pd.Series):
        # TODO: Refactor
        X_cat = X[X.columns[X.nunique()>2]].select_dtypes(include=['object', 'category'])
        # TODO: Add residual behavior
        # if self.use_residuals:
        #     residuals = y-pd.concat(base_preds).sort_index()
        curr_comp = 'full'
        best_catint_order = 1
        use_freq = {}
        for order in range(2,4): # TODO: Add max_order as parameter
            if len(X_cat.columns) >= order:
                self.scores[f'cat_int{order}'] = {}
                self.scores[f'cat_int{order}_andFreq'] = {}
                # self.scores[f'cat_int{order}_onlyFreq'] = {}

                if self.use_residuals:
                    res = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, use_filters=False)], lgb_model_type=self.lgb_model_type, residuals=residuals)
                    self.scores[f'cat_int{order}'][f'lgb-{self.lgb_model_type}'], int_preds = res['scores'], res['preds']

                    if self.target_type != 'regression':
                        self.scores[f'cat_int{order}'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])
                    # TODO: Add some additional test to not always attempt +freq
                    # I.e., only if the previous order was better with freq
                    res = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, add_freq=True, use_filters=False)], lgb_model_type=self.lgb_model_type, residuals=residuals)
                    self.scores[f'cat_int{order}_andFreq'][f'lgb-{self.lgb_model_type}'], intfreq_preds = res['scores'], res['preds']
                    if self.target_type != 'regression':
                        self.scores[f'cat_int{order}_andFreq'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in intfreq_preds])
                else:
                    res = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, use_filters=False)], lgb_model_type=self.lgb_model_type)
                    self.scores[f'cat_int{order}'][f'lgb-{self.lgb_model_type}'], int_preds = res['scores'], res['preds']
                    
                    res = self.get_lgb_performance(X, y, custom_prep=[CatIntAdder(self.target_type, max_order=order, add_freq=True, use_filters=False)], lgb_model_type=self.lgb_model_type)
                    self.scores[f'cat_int{order}_andFreq'][f'lgb-{self.lgb_model_type}'], intfreq_preds = res['scores'], res['preds']

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

    def detect_num_interactions(self, X: pd.DataFrame, y: pd.Series, base_preds: pd.Series, candidate_cols: List[str] = None):
        X_num = X[X.columns[X.nunique()>2]].select_dtypes(exclude=['object', 'category'])

        if base_preds is not None:
            residuals = y - base_preds
        else:
            residuals = None
        
        curr_comp = 'full'
                        
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

                # TODO: Get rid of code dupe and move parts to detect_general
                if self.use_residuals:
                    res = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=self.lgb_model_type, residuals=residuals)
                    self.scores[f'num_int{order}'][f'lgb-{self.lgb_model_type}'], int_preds = res['scores'], res['preds']
                    if self.target_type != 'regression':
                        self.scores[f'num_int{order}'][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in int_preds])                    
                else:
                    res = self.get_lgb_performance(X, y, custom_prep=[preprocessor], lgb_model_type=self.lgb_model_type)
                    self.scores[f'num_int{order}'][f'lgb-{self.lgb_model_type}'], int_preds = res['scores'], res['preds']

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

    def detect_bin_summarize(self, X: pd.DataFrame, y: pd.Series, base_preds: pd.Series, base_feature_importances: List[pd.Series]):
        # TODO: Test whether this even makes sense
        if self.use_residuals:
            residuals = y - base_preds
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
                # print(f"\rBinary FE: {len(use_cols)}/{imp_df.index[i:]} columns processed, current performance={curr_ll:.4f}, new performance={new_ll:.4f}", end="", flush=True)
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
        self.scores['bin-sum'][f'lgb-{self.lgb_model_type}'] = self.get_lgb_performance(X_new, y, lgb_model_type=self.lgb_model_type)['scores']

    def detect_general(self, X: pd.DataFrame, y: pd.Series, get_technique: Callable, technique_name: str, add_to_running: bool = False, base_preds: pd.Series = None):
        '''
        Currently tested for cat_as_num, cat_freq, 
        '''
        # TODO: Add option to continue training from base preds instead of using residuals
        if base_preds is not None:
            residuals = y - base_preds
        else:
            residuals = None

        self.scores[technique_name] = {}
        custom_prep = self.preprocessors_running + [get_technique()]
        res = self.get_lgb_performance(X, y, lgb_model_type=self.lgb_model_type, custom_prep=custom_prep, residuals=residuals)
        self.scores[technique_name][f'lgb-{self.lgb_model_type}'], preds = res['scores'], res['preds']

        if self.target_type != 'regression' and base_preds is not None:
            self.scores[technique_name][f'lgb-{self.lgb_model_type}'] = np.array([self.scorer(y.iloc[y_pred.index],y_pred) for y_pred in preds])

        if np.mean(self.scores[technique_name][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
            if add_to_running:
                self.base_transformers.append(get_technique().fit(X))
                self.preprocessors_running.append(get_technique())
            else:
                self.transformers.append(get_technique().fit(X, y))

    def fit(self, X_input: pd.DataFrame, y_input: pd.Series = None):
        '''
        When implementing a new engineering technique, please ensure that it is properly integrated into the following functions:
        - registered_techniques: Ensure that the technique is added to the list 
        - filter_techniques: Ensure that the technique is only applied when applicable to the dataset
        - adjust_detection_order: Ensure that the technique is applied in a sensible order (e.g., drop_irrelevant should be first)
        - The actual detection function (e.g., detect_cat_interactions)
        - The actual transformer (e.g., CatIntAdder)

        '''
        # TODO: Add functionality to resume fitting with additional preprocessors, after the model was fitted once
        # Ensure correct data format
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)
        if not isinstance(y_input, pd.Series):
            y_input = pd.Series(y_input)
        X = X_input.copy()
        y = y_input.copy()

        y = adjust_target_format(y=y, target_type = self.target_type) 

        # TODO: Verify that this block is truly not needed, also in base preprocessors functions
        # Save some properties of the original input data
        # self.orig_dtypes = {col: "categorical" if dtype in ["object", "category", str] else "numeric" for col,dtype in dict(X.dtypes).items()}
        # for col in self.orig_dtypes:
        #     if X[col].nunique()<=2:
        #         self.orig_dtypes[col] = "binary"
        # self.original_cat_features = [key for key, value in self.orig_dtypes.items() if value == "categorical"]
        # self.col_names = X.columns
        # self.changes_to_cols = {col: "None" for col in self.col_names}
        # self.changes_to_cols = {col: "raw" for col in self.col_names} 

        # Apply basic steps from AG preprocessing 
        X = self.prefilter_X(X)

        # Check which of the specified engineering techniques are applicable to the dataset
        # FIXME: Need a global function to filter to engineering techniques that will be applied and only fit the full model if necessary
        applicable_techniques = self.filter_techniques(X, y)
        
        if len(applicable_techniques) > 0:
            applicable_techniques = self.adjust_detection_order(applicable_techniques)
            
            # TODO: Check whether preds & base importance is always needed
            base_feature_importances, base_preds_lst = self.get_base_scores(X, y) 

            if self.use_residuals:
                base_preds = pd.concat(base_preds_lst).sort_index()
            else:
                base_preds = None

        skip_techniques = []
        for t in applicable_techniques:
            # TODO: Make add_to_running a global hyperparameter and adjust functions that would change the data to be feature adding functions instead
            if t in skip_techniques:
                continue
            if self.verbose:
                print(f"Test engineering technique: {t}")
            if t == 'drop_irrelevant':
                # TODO: Consider replacing with AG method here
                self.irrelevant_feature_importances = pd.DataFrame(base_feature_importances).mean().sort_values()
                # TODO: Add as a preprocessor
                self.drop_cols.extend(self.irrelevant_feature_importances[self.irrelevant_feature_importances < self.technique_params['drop_irrelevant']['irrelevant_threshold']].index.tolist())
                X = X.drop(columns=self.drop_cols, errors='ignore', axis=1)
            elif t == 'quantile_reg':
                self.detect_quantile_regression(X, y)
            elif t == 'linear_residuals':
                self.detect_linear_residuals(X, y)
            elif t == 'duplicate_mapping':
                self.detect_duplicate_mapping(X, y)
            elif t == 'cat_as_num':
                self.detect_general(X, y, lambda: CatAsNumTransformer(), 'cat_as_num', add_to_running=True, base_preds=base_preds)
                if np.mean(self.scores[t][f'lgb-{self.lgb_model_type}']) > np.mean(self.scores['full'][f'lgb-{self.lgb_model_type}']):
                    # Remove all further cat engineering techniques after 'cat_as_num'
                    cat_techniques = ['cat_freq', 'cat_as_loo', 'cat_as_ohe', 'cat_int', 'cat_groupby']
                    idx = applicable_techniques.index('cat_as_num')
                    # Skip cat techniques after 'cat_as_num' if conversion happens
                    skip_techniques.extend([t for t in applicable_techniques[idx+1:] if t in cat_techniques])
            elif t == 'cat_freq':
                # TODO: avoid testing for distinctive cat freq twice (currently also done during technique filtering). Could be done by saving candidate_cols and explicitly providing them
                self.detect_general(X, y, lambda: FrequencyEncoder(), 'cat_freq', add_to_running=False, base_preds=base_preds)
            # TODO: Properly integrate cat_as_loo and cat_as_ohe
            elif t == 'cat_as_loo':
                self.detect_general(X, y, lambda: CatLOOTransformer(), 'cat_as_loo', add_to_running=False, base_preds=base_preds)
            elif t == 'cat_as_ohe':
                # TODO: Might consider using from tabprep.preprocessors import FrequencyOHE

                self.detect_general(X, y, lambda: OneHotPreprocessor(), 'cat_as_ohe', add_to_running=False, base_preds=base_preds)
            elif t == 'SVD':
                self.detect_general(X, y, lambda: SVDPreprocessor(), 'SVD', add_to_running=False, base_preds=base_preds)
            elif t == 'cat_int':
                self.detect_cat_interactions(X, y)
            elif t == 'cat_groupby':
                self.detect_general(X, y, lambda: CatGroupByAdder(min_cardinality=self.min_cardinality), 'cat_groupby', add_to_running=False, base_preds=base_preds)
            elif t == 'num_int':
                imp_df = pd.DataFrame(base_feature_importances).mean()
                num_cols = X[X.columns[X.nunique()>2]].select_dtypes(exclude=['object', 'category']).columns.tolist()
                candidate_cols = imp_df.loc[num_cols].sort_values(ascending=False).head(100).index.tolist()
                self.detect_num_interactions(X, y, base_preds=base_preds, candidate_cols=candidate_cols)
            elif t == 'groupby':
                prep_params = {
                    'target_type': self.target_type,
                    'min_cardinality': self.min_cardinality,
                    'use_mvp': False,
                    'mean_difference': True,
                    'num_as_cat': False,
                }
                self.detect_general(X, y, lambda: GroupByFeatureEngineer(**prep_params), 'groupby', add_to_running=False, base_preds=base_preds)
            elif t == 'bin_summarize':
                self.detect_bin_summarize(X, y, base_preds=base_preds, base_feature_importances=base_feature_importances)
            elif t == 'duplicate_count':
                self.detect_general(X, y, lambda: DuplicateCountAdder(), 'duplicate_count', add_to_running=False, base_preds=base_preds)
            elif t == 'linear_feature':
                self.detect_general(X, y, lambda: LinearFeatureAdder(self.target_type, linear_model_type='lasso'), 'linear_feature', add_to_running=False, base_preds=base_preds)
            else:
                raise ValueError(f"Engineering technique {t} not recognized. Available techniques: {self.registered_techniques}")

        # TODO: Add rank, min, max to GroupBy interactions
        # TODO: Add technique to generate few but higher-order feature interactions
        # TODO: Add technique to find binary interactions

        """
        Returns
        -------
        self : AllInOneEngineer
            The fitted instance of AllInOneEngineer.
        """
        return self

    def transform(self, X_input: pd.DataFrame):
        """
         Applies feature engineering transformations to the input DataFrame.
         - base_transformers: sequentially modify the original features (e.g., encoding, replacements).
         - transformers: add new features to the dataset (e.g., interaction features).
        """
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)
        X = X_input.copy()
        
        # Drop columns identified as irrelevant
        X = X.drop(self.drop_cols, axis=1, errors='ignore')

        # Apply base_transformers that modify the original features
        for transformer in self.base_transformers:
            X = transformer.transform(X)

        # Apply transformers that add new features
        new_feats = []
        for transformer in self.transformers:
            X_feat = transformer.transform(X)
            # Only add new columns generated by the transformer
            new_cols = [col for col in X_feat.columns if col not in X.columns]
            new_feats.append(X_feat[new_cols])

        return pd.concat([X, *new_feats], axis=1)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the AllInOneEngineer to the data and then transforms it.
        """
        self.fit(X, y)
        return self.transform(X)    
