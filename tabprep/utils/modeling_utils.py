import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from collections import Counter

from sklearn.preprocessing import QuantileTransformer, StandardScaler, PowerTransformer

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss
from tabprep.proxy_models import CustomLinearModel
from tabprep.utils.misc import assign_closest

from typing import Tuple, List, Callable, Literal

def clean_feature_names(X_input: pd.DataFrame) -> pd.DataFrame:
    '''
    Required to avoid issues with special characters in column names for some libraries (like LightGBM).
    Replaces: '[', ']', ':', '<', '>', '=', ',', with '' and ' ' with '_'.
    
    Example: 'feature[1]:value' -> 'feature1_value'

    Args:
        X_input (pd.DataFrame): Input dataframe with original column names.
    Returns:
        pd.DataFrame: Dataframe with cleaned column names.
    '''    
    X = X_input.copy()
    X.columns = [str(col).replace('[', '').replace(']', '').replace(':', '')
                                            .replace('<', '').replace('>', '')
                                            .replace('=', '').replace(',', '')
                                            .replace(' ', '_') for col in X.columns]
    
    return X

def clean_series(x: pd.Series) -> pd.Series:
    '''
    Cleans a pandas Series by handling special string patterns and converting to float.
    Specifically, it processes strings that represent inequalities (e.g., '<5', '>10').
    
    Args:
        x (pd.Series): Input series with potential special string patterns.
    Returns:
        pd.Series: Cleaned series with float values.

    '''
    def parse_value(val):
        if pd.isnull(val):
            return np.nan
        s = str(val).strip()
        
        # Handle '<X' or '>X' patterns
        if re.match(r'^<\s*\d+[.,]?\d*$', s):
            num = re.sub(r'[^\d.,]', '', s)
            try:
                return float(num.replace(',', '')) - 1
            except:
                return np.nan
        elif re.match(r'^>\s*\d+[.,]?\d*$', s):
            num = re.sub(r'[^\d.,]', '', s)
            try:
                return float(num.replace(',', '')) + 1
            except:
                return np.nan
        else:
            # Default cleaning: remove all non-digit/dot/comma characters
            cleaned = re.sub(r'[^\d.,]', '', s)
            try:
                return float(cleaned.replace(',', ''))
            except:
                return np.nan

    return x.apply(parse_value)

def adjust_target_format(
        y: pd.Series, 
        target_type: Literal['binary', 'multiclass', 'regression'], 
        multi_as_bin: bool = False
        ) -> pd.Series:
    '''
    Adjusts the target variable format based on the specified target type.
    - For 'binary': Converts to binary (0 and 1) using the most frequent class.
    - For 'multiclass': Converts to integer labels (0, 1, 2, ...) using LabelEncoder. If multi_as_bin is True, converts to binary using the most frequent class.
    - For 'regression': Converts to float.

    Args:
        y (pd.Series): Target variable.
        target_type (str): Type of target variable ('binary', 'multiclass', 'regression').
        multi_as_bin (bool): If True and target_type is 'multiclass', converts to binary using the most frequent class.
    Returns:
        pd.Series: Adjusted target variable.
    Raises:
        ValueError: If target_type is not one of 'binary', 'multiclass', 'regression'.
    '''
    y_out = y.copy()
    if target_type == "binary":
        y_out = (y==y.value_counts().index[0]).astype(int)        
    elif target_type == "multiclass" and multi_as_bin:
        y_out = (y==y.value_counts().index[0]).astype(int)        
    elif target_type == "multiclass" and not multi_as_bin:
        y_out = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=y.name)
    elif target_type == "regression":
        y_out = y.astype(float)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return y_out

def adapt_lgb_params(
        target_type: Literal['binary', 'multiclass', 'regression'],
        lgb_model_type: str = 'default',
        X: pd.DataFrame = None, 
        y: pd.Series = None, 
        base_params: dict = None, 
        ) -> dict:
    # TODO: This function is outdated and likely can be removed.
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
            "objective": "binary" if target_type=="binary" else "regression",
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
        raise ValueError(f"Unknown lgb_model_type: {lgb_model_type}. Use 'unique-based', 'high-capacity', 'high-capacity-binned', 'full-capacity' or 'default'.")
    
    return params

def make_cv_function(
        target_type: Literal['binary', 'multiclass', 'regression'],
        n_folds: int = 5,  
        random_state: int = 42,
        vectorized: bool = False,
        verbose: bool = False,
        groups: np.ndarray | None = None,
        drop_modes: int = 0
    ) -> Callable:
    """CV creation function for vectorized versions of the TargetEncoderModels."""
    # TODO: Make a class out of that CV function generator
    import lightgbm as lgb
    if target_type=='binary':
        scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif target_type=='regression':
        scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif target_type=='multiclass':
        scorer = lambda y_true, y_pred: -log_loss(y_true, y_pred) # FIXME: Adjust to make sure function runs when not all classes are present
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:   
        raise ValueError("target_type must be 'binary' or 'regression'")

    def _density_weights(
            y_base: pd.Series, 
            n_bins: int = 20, 
            clip: Tuple[float, float] = (0.2, 5.0)
            ) -> np.ndarray:
        vals = np.asarray(y_base)
        # Continuous targets: quantile-bin inverse frequency
        if np.issubdtype(vals.dtype, np.number) and np.unique(vals).size > max(10, n_bins // 2):
            from sklearn.preprocessing import KBinsDiscretizer
            kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            bins = kb.fit_transform(vals.reshape(-1, 1)).astype(int).ravel()
            counts = np.bincount(bins, minlength=n_bins).astype(float)
            w = 1.0 / np.maximum(counts[bins], 1.0)
        else:
            # Discrete labels: class inverse frequency
            _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
            w = 1.0 / np.maximum(counts[inv], 1.0)
        w = w / np.mean(w)
        if clip is not None:
            w = np.clip(w, clip[0], clip[1])
        return w

    def cv_func(
            X_df: pd.DataFrame, 
            y_s: pd.Series, 
            pipeline: Pipeline, 
            early_stopping_rounds: int = 20,
            X_test_in: pd.DataFrame | None  = None, 
            y_test_in: pd.Series | None = None,
            custom_prep: list[type] | None = None,
            return_iterations: bool = False,
            return_preds: bool = False,
            return_importances: bool = False,
            scale_y: Literal['standard', 'power', 'quantile', 'linear_residuals', 'log', 'exp'] | None = None,
            original_y: pd.Series | None = None,
            reg_assign_closest_y: bool = False,
            sample_weights: Literal['density', 'cls_balanced', 'duplicates'] | None = None,
            init_score: Literal['linear'] | None = None,
                ):
        
        scores = []
        iterations = []
        all_preds = []
        test_scores = []
        test_preds = []
        feature_importances = []
        if original_y is not None and scale_y != 'linear_residuals':
            splits = cv.split(X_df, original_y, groups=groups)
            scorer_use = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr)
        else:
            splits = cv.split(X_df, y_s, groups=groups)
            scorer_use = scorer
        for train_idx, test_idx in splits:
            X_tr, y_tr = X_df.iloc[train_idx], y_s.iloc[train_idx]
            X_val, y_val = X_df.iloc[test_idx], y_s.iloc[test_idx]
            if X_test_in is not None and y_test_in is not None:
                X_test, y_test = X_test_in.copy(), y_test_in.copy()
            else:
                X_test, y_test = None, None

            col_names = X_df.columns

            if custom_prep is not None:
                for prep in custom_prep:
                    X_tr = prep.fit_transform(X_tr, y_tr)
                    X_val = prep.transform(X_val)
                    if X_test is not None and y_test is not None:
                        X_test = prep.transform(X_test)

                if not isinstance(X_tr, pd.DataFrame):
                    if X_tr.shape[1]==len(col_names):
                        X_tr = pd.DataFrame(X_tr, columns=X_df.columns)
                    else:
                        X_tr = pd.DataFrame(X_tr)
                if not isinstance(X_val, pd.DataFrame):
                    if X_val.shape[1]==len(col_names):
                        X_val = pd.DataFrame(X_val, columns=X_df.columns)
                    else:
                        X_val = pd.DataFrame(X_val)
                if X_test is not None and y_test is not None and not isinstance(X_test, pd.DataFrame):
                    if X_test.shape[1]==len(col_names):
                        X_test = pd.DataFrame(X_test, columns=X_df.columns)
                    else:
                        X_test = pd.DataFrame(X_test)

            if scale_y is not None:
                if scale_y in ['standard', 'power', 'quantile']:
                    if scale_y  == 'standard':
                        scaler = StandardScaler()
                    elif scale_y  == 'power':
                        scaler = PowerTransformer()
                    elif scale_y  == 'quantile':
                        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
                        
                    y_tr = pd.Series(scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel(), name=y_tr.name, index=y_tr.index)
                    y_val = pd.Series(scaler.transform(y_val.values.reshape(-1, 1)).ravel(), name=y_val.name, index=y_val.index)
                    if X_test is not None and y_test is not None:
                        y_test = pd.Series(scaler.transform(y_test.values.reshape(-1, 1)).ravel(), name=y_test.name, index=y_test.index)
                elif scale_y  == 'log':
                    y_tr = np.log1p(y_tr)
                    y_val = np.log1p(y_val)
                    if X_test is not None and y_test is not None:
                        y_test = np.log1p(y_test)
                elif scale_y  == 'exp':
                    y_tr = np.expm1(y_tr)
                    y_val = np.expm1(y_val)
                    if X_test is not None and y_test is not None:
                        y_test = np.expm1(y_test)
                elif scale_y == 'linear_residuals':
                    lm = CustomLinearModel(target_type)
                    lm.fit(X_tr, y_tr)
                    y_tr_lin = lm.predict(X_tr)
                    y_val_lin = lm.predict(X_val)
                    if X_test is not None and y_test is not None:
                        y_test_lin = lm.predict(X_test)
                else:
                    raise ValueError("scale_y must be 'standard' or 'log'")
            
            final_model = pipeline.named_steps["model"]

            if init_score == 'linear':
                def to_logit(p, eps=1e-6, shrink=0.5):
                    p = np.clip(p, eps, 1 - eps)
                    return np.log(p / (1 - p)) * shrink
                lm = CustomLinearModel(target_type)
                lm.fit(X_tr, y_tr)
                y_tr_base = lm.predict(X_tr)
                y_te_base = lm.predict(X_val)
                if target_type in ['binary', 'regression']:
                    y_tr_base = y_tr_base.ravel()
                    y_te_base = y_te_base.ravel()
                if target_type == 'binary':
                    y_tr_base = to_logit(y_tr_base, eps=0.05, shrink=0.1)  
                    y_te_base = to_logit(y_te_base, eps=0.05, shrink=0.1)  
                if X_test is not None and y_test is not None:
                    y_test_base = lm.predict(X_test)
                    if target_type in ['binary', 'regression']:
                        y_test_base = y_test_base.ravel()
                    if target_type == 'binary':
                        y_test_base = to_logit(y_test_base, eps=0.05, shrink=0.1)
            else:
                y_tr_base = None
                y_te_base = None

            if sample_weights == 'density':
                if original_y is not None:
                    y_tr_for_weights = original_y.iloc[train_idx]
                else:
                    y_tr_for_weights = y_s.iloc[train_idx]
                w_tr = _density_weights(y_tr_for_weights)                
                w_te = None
                w_test = None  
            elif sample_weights == 'cls_balanced':
                val_map = y_tr.value_counts(normalize=True).to_dict()
                w_tr = 1-y_tr.map(val_map)
                w_te = 1-y_val.map(val_map)
                if X_test is not None and y_test is not None:
                    w_test = 1-y_test.map(val_map)

            elif sample_weights == 'duplicates':
                train_str = X_tr.astype(str).sum(axis=1)
                cnt_map = dict(train_str.value_counts())
                
                X_tr = X_tr.drop_duplicates()
                w_tr = X_tr.astype(str).sum(axis=1).map(cnt_map).values
                if target_type == "regression":
                    y_tr_map = y_tr.groupby(train_str).mean().astype(float)
                elif target_type == "binary":
                    y_tr_map = y_tr.groupby(train_str).mean().round()
                # y_tr_use = y_tr.loc[X_tr.index]
                y_tr = X_tr.astype(str).sum(axis=1).map(y_tr_map)
                w_te = None
                w_test = None
            elif sample_weights is None:
                w_tr = None
                w_te = None
                w_test = None 
            else:
                raise ValueError("sample_weights must be None or 'density'")

            # if it's an LGBM model, pass in eval_set + callbacks
            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                if scale_y == 'linear_residuals':
                    # if scale_y is linear_residuals, we need to pass the residuals
                    pipeline.fit(
                        X_tr, y_tr - y_tr_lin,
                        **{
                            "model__eval_set": [(X_val, y_val - y_val_lin)],
                            "model__callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                            "model__sample_weight": w_tr,
                            "model__eval_sample_weight": [w_te],
                            "model__init_score": y_tr_base,
                            "model__eval_init_score": [y_te_base],
                            # "model__verbose": False
                        }
                    )
                else:
                    pipeline.fit(
                        X_tr, y_tr,
                        **{
                            "model__eval_set": [(X_val, y_val)],
                            "model__callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                            "model__sample_weight": w_tr,
                            "model__eval_sample_weight": [w_te],
                            "model__init_score": y_tr_base,
                            "model__eval_init_score": [y_te_base],
                            # "model__verbose": False
                        }
                    )
            else:
                # dummy / other estimators: plain fit
                pipeline.fit(X_tr, y_tr)

            # predict
            if target_type == "regression" or scale_y == 'linear_residuals' or not hasattr(pipeline, "predict_proba"):
                val_preds = pipeline.predict(X_val)
                if X_test is not None and y_test is not None:
                    preds_test = pipeline.predict(X_test)
            elif target_type == "binary":
                if vectorized:
                    val_preds = pipeline.predict_proba(X_val)[:, :, 1]
                else:
                    val_preds = pipeline.predict_proba(X_val)[:, 1]
                if X_test is not None and y_test is not None:
                    preds_test = pipeline.predict_proba(X_test)[:, 1]
            elif target_type == 'multiclass':
                if vectorized:
                    val_preds = pipeline.predict_proba(X_val)
                else:
                    val_preds = pipeline.predict_proba(X_val)
                if X_test is not None and y_test is not None:
                    preds_test = pipeline.predict_proba(X_test)

            if scale_y is not None:
                if scale_y in ['standard', 'power', 'quantile']:
                    val_preds = pd.Series(scaler.inverse_transform(val_preds.reshape(-1, 1)).ravel(), name=y_val.name, index=y_val.index)
                    y_tr = pd.Series(scaler.inverse_transform(y_tr.values.reshape(-1, 1)).ravel(), name=y_tr.name, index=y_tr.index)
                    y_val = pd.Series(scaler.inverse_transform(y_val.values.reshape(-1, 1)).ravel(), name=y_val.name, index=y_val.index)
                    if X_test is not None and y_test is not None:
                        y_test = pd.Series(scaler.inverse_transform(y_test.values.reshape(-1, 1)).ravel(), name=y_test.name, index=y_test.index)
                elif scale_y == 'log':
                    val_preds = np.expm1(val_preds)
                    y_tr = np.expm1(y_tr)
                    y_val = np.expm1(y_val)
                    if X_test is not None and y_test is not None:
                        y_test = np.expm1(y_test)
                elif scale_y  == 'exp':
                    val_preds = np.log1p(val_preds)
                    y_tr = np.log1p(y_tr)
                    y_val = np.log1p(y_val)
                    if X_test is not None and y_test is not None:
                        y_test = np.log1p(y_test)
                elif scale_y == 'linear_residuals':
                    if target_type == 'binary':
                        val_preds = val_preds + y_val_lin
                        val_preds = val_preds.clip(0.00001,0.9999)
                        if X_test is not None and y_test is not None:
                            preds_test = val_preds + y_test_lin
                            preds_test = preds_test.clip(0.00001,0.9999)

                else:
                    raise ValueError("scale_y must be 'standard' or 'log'")

            if reg_assign_closest_y and target_type == 'regression':
                # Assign closest y value to y_tr and y_val
                val_preds = assign_closest(val_preds,y_tr)
                if X_test is not None and y_test is not None:
                    preds_test = assign_closest(preds_test,y_tr)


            if vectorized:
                scores.append({col: scorer_use(y_val, val_preds[:,num]) for num, col in enumerate(X_df.columns)})
            else:
                scores.append(scorer_use(y_val, val_preds))

            if X_test is not None and y_test is not None:
                test_scores.append(scorer_use(y_test, preds_test))

            if return_preds:
                if isinstance(final_model, lgb.LGBMClassifier) and y_tr.nunique() > 2:
                    all_preds.append(pd.DataFrame(val_preds, index=y_val.index))
                else:
                    all_preds.append(pd.Series(val_preds, name=y_val.name, index=y_val.index))
                if X_test is not None and y_test is not None:
                    if isinstance(final_model, lgb.LGBMClassifier) and y_tr.nunique() > 2:
                        test_preds.append(pd.DataFrame(preds_test, index=y_test.index))
                    else:
                        test_preds.append(pd.Series(preds_test, name=y_test.name, index=y_test.index))

            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                iterations.append(pipeline.named_steps['model'].booster_.num_trees())
                feature_importances.append(
                    pd.Series({col: imp for col,imp in zip(X_tr.columns,pipeline.named_steps['model'].feature_importances_)})
                )
            elif hasattr(pipeline.named_steps['model'], "coef_"):
                if len (X_tr.columns)>1:
                    feature_importances.append(
                        pd.Series({col: imp for col,imp in zip(X_tr.columns,pipeline.named_steps['model'].coef_)})
                    )
                else:
                    feature_importances.append(
                        pd.Series({X_tr.columns[0]: pipeline.named_steps['model'].coef_[0]})
                    )

        # TODO: Might change to return a dict instead
        return_dict = {'scores': np.array(scores)}
        if X_test is not None and y_test is not None:
            return_dict['test_scores'] = np.array(test_scores)
        if return_iterations:
            return_dict['iterations'] = iterations
        if return_preds:
            return_dict['preds'] = all_preds
            if X_test is not None and y_test is not None:
                return_dict['test_preds'] = test_preds
        if return_importances:
            return_dict['importances'] = feature_importances

        return return_dict

    return cv_func

def safe_stratified_group_kfold(X, y, groups, n_splits=5, max_attempts=1, random_state=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    attempt = 0

    while attempt < max_attempts:
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            test_labels = y[test_idx]
            class_counts = Counter(test_labels)
            if len(class_counts) < 2:
                break  # This fold has only one class
        else:
            return sgkf#.split(X, y, groups)  # All folds are good
        
        attempt += 1

    print("Could not generate stratified group folds with both classes in all test sets.")
    return None  # If we reach here, it means we couldn't find a valid split