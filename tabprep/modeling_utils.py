import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Literal

def adjust_target_format(
        y: pd.Series, 
        target_type: Literal['binary', 'multiclass', 'regression'], 
        multi_as_bin: bool = False
        ):
    y_out = y.copy()
    if target_type == "binary":
        y_out = (y==y.value_counts().index[0]).astype(int)        
    elif target_type == "multiclass" and multi_as_bin:
        y_out = (y==y.value_counts().index[0]).astype(int)        
    elif target_type == "multiclass" and not multi_as_bin:
        y_out = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=y.name)
    else:
        y_out = y.astype(float)

    return y_out

def adapt_lgb_params(
        target_type: Literal['binary', 'multiclass', 'regression'],
        lgb_model_type: str = 'default',
        X: pd.DataFrame = None, 
        y: pd.Series = None, 
        base_params: dict = None, 
        ):
    
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