import openml
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from traitlets import default
from utils import *
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import os
import pickle
import openml
from ft_detection import FeatureTypeDetector
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

def get_config_result(did, tid, dataset_variants: list=['openml'], ftd_params: dict={}):
    res = {}
    
    # Get dataset
    data = openml.datasets.get_dataset(did)
    X, _, _, _ = data.get_data() # TODO: Adapt to use tasks instead
    y = X[data.default_target_attribute]
    X = X.drop(columns=[data.default_target_attribute])
    
    # Get task type
    task_type = openml.tasks.get_task(tid).task_type
    if task_type == "Supervised Classification" and y.nunique() == 2:
        target_type = "binary"
    elif task_type == "Supervised Classification":
        target_type = "multiclass"
    elif task_type == "Supervised Regression":
        target_type = "regression"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    res['target_type'] = target_type
    res["dataset_name"] = data.name
    res["did"] = did
    res["tid"] = tid

    if 'openml' in dataset_variants:    
        res['openml'] = {}
        
        ftd = FeatureTypeDetector(target_type=target_type, **ftd_params)
        ftd.fit(X, y, verbose=False)
        
        res['openml']['new_categorical'] = list(ftd.cat_dtype_maps.keys())
        res['openml']['new_numeric'] = [col for col in X.columns if ftd.dtypes[col]=="numeric" and ftd.orig_dtypes[col]!="numeric"]
        res['openml']['dtypes'] = ftd.dtypes
        res['openml']['orig_dtypes'] = ftd.orig_dtypes
        res["openml"]['scores'] = ftd.scores
        res['openml']['significances'] = ftd.significances
        res['openml']['params'] = ftd.get_params()


    if 'AG' in dataset_variants:
        feature_generator = AutoMLPipelineFeatureGenerator()
        feature_generator.fit(X)
        X_ag = feature_generator.transform(X.copy())

        if X_ag.shape != X.shape or any(X.dtypes.sort_index() != X_ag.dtypes.sort_index()):
            res['AG'] = {}
            
            ftd = FeatureTypeDetector(target_type=target_type, **ftd_params)
            ftd.fit(X_ag, y, verbose=False)
            
            res['AG']['new_categorical'] = list(ftd.cat_dtype_maps.keys())
            res['AG']['new_numeric'] = [col for col in X_ag.columns if ftd.dtypes[col]=="numeric" and ftd.orig_dtypes[col]!="numeric"]
            res['AG']['dtypes'] = ftd.dtypes
            res['AG']['orig_dtypes'] = ftd.orig_dtypes
            res["AG"]['scores'] = ftd.scores
            res['AG']['significances'] = ftd.significances
            res['AG']['params'] = ftd.get_params()
        elif 'openml' in dataset_variants:  
            res['AG'] = res["openml"]
        else:
            res['AG'] = None
    if 'csv' in dataset_variants:
        X.to_csv(f"dataset_{did}_tid_{tid}.csv", index=False)
        X_csv = pd.read_csv(f"dataset_{did}_tid_{tid}.csv")
        os.remove(f"dataset_{did}_tid_{tid}.csv")

        if any(X_csv!=X):
            res['csv'] = {}
            
            ftd = FeatureTypeDetector(target_type=target_type, **ftd_params)
            ftd.fit(X_csv, y, verbose=False)

            res['csv']['new_categorical'] = list(ftd.cat_dtype_maps.keys())
            res['csv']['new_numeric'] = [col for col in X_csv.columns if ftd.dtypes[col]=="numeric" and ftd.orig_dtypes[col]!="numeric"]
            res['csv']['dtypes'] = ftd.dtypes
            res['csv']['orig_dtypes'] = ftd.orig_dtypes
            res["csv"]['scores'] = ftd.scores
            res['csv']['significances'] = ftd.significances
            res['csv']['params'] = ftd.get_params()

        elif 'openml' in dataset_variants:
            res['AG'] = res["openml"]
        else:
            res['AG'] = None

    return res

if __name__ == "__main__":
    exp_name = "ftd_variants"
    dataset_variants = ['openml', 'AG', 'csv']
    ftd_params_lst = [
        ('default', {}),
        ('cardinality10', {'min_q_as_num': 11}),
        ('10folds', {'n_folds': 10}),
        ('default_lgb', {"lgb_model_type": 'default'}),
        ('high_corr_1', {"use_highest_corr_feature": True, 'num_corr_feats_use': 1}),
        ('high_corr_5', {"use_highest_corr_feature": True, 'num_corr_feats_use': 5}),
    ]

    if os.path.exists(f"{exp_name}.pkl"):
        with open(f"{exp_name}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {}
    collection_id = 457  # TabArena
    # collection_id = 334 # Tabular benchmark categorical classification
    # collection_id = 335 # Tabular benchmark categorical regression
    # collection_id = 336 # Tabular benchmark numerical regression
    # collection_id = 337 # Tabular benchmark numerical classification
    # collection_id = 379 # TabZilla-hard
    # Fetch the benchmark suite
    benchmark_suite = openml.study.get_suite(collection_id)
    dids = benchmark_suite.data
    tids = benchmark_suite.tasks

    for d_num, (did, tid) in enumerate(zip(dids, tids)):
        data = openml.datasets.get_dataset(did)
        dataset_name = data.name
        if dataset_name in ['superconductivity']:
            continue
        if dataset_name not in results:
            results[dataset_name] = {}
        print(dataset_name)
        for i, ftd_params_tuple in enumerate(ftd_params_lst):
            ftd_params_name, ftd_params = ftd_params_tuple
            if ftd_params_name not in results[dataset_name]:
                print(f"Running {ftd_params_name} for {dataset_name}...")
                res = get_config_result(did=did, tid=tid, dataset_variants=dataset_variants, ftd_params=ftd_params)
                results[dataset_name][ftd_params_name] = res
            else:
                print(f"Skipping {ftd_params_name} for {dataset_name} as it already exists.")

        experiments = list(results[dataset_name].keys())
        dataset_variants = ['openml', 'AG', 'csv']

        for exp in experiments:
            print(f"Experiment: {exp}")
            print("New categoricals: ", {v: results[dataset_name][exp][v]['new_categorical'] for v in dataset_variants})
            print("New numerics: ", {v: results[dataset_name][exp][v]['new_numeric'] for v in dataset_variants})
            print("--" * 20)

        # if d_num==2:
        #     break

        with open(f"{exp_name}.pkl", "wb") as f:
            pickle.dump(results, f)

