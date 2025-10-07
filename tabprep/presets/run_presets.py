import pandas as pd
import numpy as np
import os
import openml

from sklearn.metrics import roc_auc_score, root_mean_squared_error

from tabprep.presets.preset_registry import LGBPresetRegistry
from tabprep.utils.modeling_utils import make_cv_function, adjust_target_format
from tabprep.utils.eval_utils import get_benchmark_dataIDs
from tabprep.proxy_models import CustomLinearModel, CustomKNNModel

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
from tabrepo.benchmark.task.openml import OpenMLTaskWrapper
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.benchmark.models.wrapper.AutoGluon_class import AGSingleBagWrapper
import pickle

from typing import List, Union, Tuple, Literal

# TODO: Check where I have the function for loading data
dataset_name = 'diabetes'
def get_openml_task(dataset_name):
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    for benchmark in ['TabArena']: # ["Grinsztajn", "TabArena", "TabZilla"]:
        tids, dids = get_benchmark_dataIDs(benchmark) 
        for tid, did in zip(tids, dids):
            # TODO: Make more efficient
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

            y = adjust_target_format(y, target_type)

    return X, y, target_type

# TODO: Check where I have the function for loading data
dataset_name = 'diabetes'
def get_dataset(dataset_name):
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
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

            y = adjust_target_format(y, target_type)

    return X, y, target_type

def filter_presets(presets, target_type, X, y):
    if target_type in ['binary', 'multiclass']:
        remove_presets = ['1NN_residuals']
        presets = [preset for preset in presets if preset not in remove_presets]
    if target_type == 'multiclass':
        remove_presets = [i for i in presets if 'linear_residual' in i] + ['duplicate_sample_loo']
        presets = [preset for preset in presets if preset not in remove_presets]
    if X.shape[1]>120:
        remove_presets = [i for i in presets if 'all_as_cat_append_w_int2' in i]
        presets = [preset for preset in presets if preset not in remove_presets]
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    cat_cardinality = X[cat_cols].nunique().sum()

    if len(cat_cols)>10 and cat_cardinality > 50000: # TODO: NEED TO ADAPT CHANGE
        remove_presets = ['cat_fe', 'cat_fe_nocatreg', 'custom_onehot', 'custom_onehot_append', 'linear_tree_standardized_withCatOHE', "linear_tree_catOHE"]
        presets = [preset for preset in presets if preset not in remove_presets]

    # if len(cat_cols)>20:
    #     remove_presets = [i for i in presets if 'cat_groupby_min_cardinality_3' in i]
    #     presets = [preset for preset in presets if preset not in remove_presets]

    if target_type != 'multiclass':
        remove_presets = [i for i in presets if i in ['multiclass_many', 'multiclass_imbalanced']]
        presets = [preset for preset in presets if preset not in remove_presets]
    
    if target_type != 'regression':
        remove_presets = [i for i in presets if i in ['quantile_levels', 'quantile_reg', 'robust_regression', 'multimodal_regression']]
        presets = [preset for preset in presets if preset not in remove_presets]

    if target_type != 'binary':
        remove_presets = [i for i in presets if i in ['calibrated_probs', 'imbalanced']]
        presets = [preset for preset in presets if preset not in remove_presets]

    if target_type == 'multiclass':
        remove_presets = [i for i in presets if i in ['cat_loo_append', 'cat_loo_replace', 'cat_loo_append_withNumeric_append', 'cat_loo_replace_withNumeric_append', 
                                                      'standardized_catloo', "standardized_TE", "standardized_distanceweights_20neighbors_catloo",
                                                      "quantile_distanceweights_20neighbors_catloo"
                                                      ]]
        presets = [preset for preset in presets if preset not in remove_presets]

    if X.shape[0]>=99000:
        remove_presets = ['arithmetic_interactions_maxorder3_5000feats']
        presets = [preset for preset in presets if preset not in remove_presets]

    # FIXME: There is some bug currently, probably simple but no time atm
    if 'fiber_ID' in X.columns:
        remove_presets = ['arithmetic_interactions_maxorder3_2000feats_noregularization']
        presets = [preset for preset in presets if preset not in remove_presets]
    # FIXME: Fix bug with customer_satisfaction_in_airline
    if 'ArrivalDelayinMinutes' in X.columns:
        remove_presets = ['RBFPCA', '2-PolynomialPCA', '3-PolynomialPCA', 'SigmoidPCA', 'RBFPCA-onlynum', '2-PolynomialPCA-onlynum', '3-PolynomialPCA-onlynum', 'SigmoidPCA-onlynum']
        presets = [preset for preset in presets if preset not in remove_presets]
    # FIXME: Fix bug with GiveMeSomeCredit
    if 'RevolvingUtilizationOfUnsecuredLines' in X.columns:
        remove_presets = ['RBFPCA', '2-PolynomialPCA', '3-PolynomialPCA', 'SigmoidPCA', 'RBFPCA-onlynum', '2-PolynomialPCA-onlynum', '3-PolynomialPCA-onlynum', 'SigmoidPCA-onlynum']
        presets = [preset for preset in presets if preset not in remove_presets]

    if X.shape[0]>30000 and X.shape[1]>120:
        remove_presets = ['3-PolynomialPCA', '2-PolynomialPCA', '3-PolynomialPCA-onlynum', '2-PolynomialPCA-onlynum']
        presets = [preset for preset in presets if preset not in remove_presets]

    return presets


def run_preset(
        X: pd.DataFrame, 
        y: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        target_type: str, 
        init_params: dict, 
        preprocessors: list, 
        cv_params: dict, 
        cv_type: str = 'custom',
        model_name: str = 'GBM'
        ):

            if target_type in ['binary', 'multiclass']:
                lgb_model = LGBMClassifier
            else:
                lgb_model = LGBMRegressor

            X_new = X.copy()
            y_new = y.copy()
            y_test_new = y_test.copy() if y_test is not None else None
            num_cols = X_new.select_dtypes(include=[np.number]).columns
            X_new[num_cols] = X_new[num_cols].astype(float)
            if X_test is not None:
                X_test_new = X_test.copy()
                X_test_new[num_cols] = X_test_new[num_cols].astype(float)
            
            obj_cols = X_new.select_dtypes(include=['object']).columns
            X_new[obj_cols] = X_new[obj_cols].astype('category')
            if X_test is not None:
                for col in obj_cols:
                    X_test_new[col] = X_test_new[col].astype(X_new[col].dtype)
            
            if cv_type == 'custom':
                cv_function = make_cv_function(target_type=target_type, verbose=False)
                res = cv_function(X_new, y_new, 
                                Pipeline([('model', lgb_model(**init_params))]), 
                                X_test_in=X_test_new, y_test_in=y_test_new,
                                custom_prep=preprocessors,
                                return_preds=True, 
                                return_importances=True,
                                return_iterations=True,  
                                **cv_params,
                                # sample_weights=None, 
                                # scale_y=None, # 'quantile, linear_residuals
                        )
                dat_results = {
                    'init_params': init_params,
                    'preprocessors': [i.__name__ for i in preprocessors],
                    'val_scores': res['scores'],
                    'mean_val_score': np.mean(res['scores']),
                    'val_preds': res['preds'],
                    'iterations': res['iterations'],
                    'feature_importances': pd.concat(res['importances'], axis=1).mean(axis=1).sort_values(ascending=False)
                }
                if 'test_scores' in res:
                    dat_results['test_scores'] = res['test_scores']
                    dat_results['mean_test_scores'] = np.mean(res['test_scores'])
            elif cv_type == 'tabarena':
                if 'fit_kwargs' in cv_params:
                    fit_kwargs = cv_params['fit_kwargs']
                else:
                    fit_kwargs = {'num_bag_folds': 8}

                if 'fit_kwargs' in cv_params:
                    fit_kwargs = cv_params['fit_kwargs']
                else:
                    fit_kwargs = {'num_bag_folds': 8}
                
                post_predict_duplicate_mapping = False
                if 'map_duplicates' in cv_params:
                    n_X_duplicates = X.duplicated().mean()
                    n_Xy_duplicates = pd.concat([X,y],axis=1).duplicated().mean()

                    # if n_X_duplicates<=0.001:
                    #     post_predict_duplicate_mapping = False
                    # elif target_type == 'multiclass':
                    #     post_predict_duplicate_mapping = False
                    if isinstance(cv_params['map_duplicates'], float):
                        if np.abs(n_X_duplicates - n_Xy_duplicates)<cv_params['map_duplicates']:
                            post_predict_duplicate_mapping = True
                        # else:
                        #     post_predict_duplicate_mapping = False
                    elif cv_params['map_duplicates'] == 'exact' and n_X_duplicates == n_Xy_duplicates:
                        post_predict_duplicate_mapping = True
                    elif cv_params['map_duplicates'] == 'all':
                        post_predict_duplicate_mapping = True
                    # else:
                    #     post_predict_duplicate_mapping = False

                # TODO: Make early stopping work with custom CV again
                # if 'early_stopping_rounds' in cv_params:
                #     init_params.update({'ag.early_stop': cv_params['early_stopping_rounds']})

                # TODO: Find a better solution for adding linear residuals
                # TODO: Decide whether we want that before or after the preprocessing
                if 'linear_residuals' in cv_params and cv_params['linear_residuals']:
                    # y_all = adjust_target_format(pd.concat([y_new,y_test_new]), target_type)
                    # y_new = y_all.loc[y_new.index]
                    # y_test_new = y_all.loc[y_test_new.index]
                    if target_type == 'binary':
                        y_test_new = (y_test_new==y_new.unique()[1]).astype(int)
                        y_new = (y_new==y_new.unique()[1]).astype(int)
                    lm = CustomLinearModel(target_type)
                    lm.fit(X, y_new)
                    y_lin = lm.predict(X)
                    # y_val_lin = lm.predict(X_val)
                    y_test_lin = lm.predict(X_test)
                    eval_metric_name = 'rmse'
                    problem_type = 'regression' 
                    init_params.update({'objective': 'regression'})
                elif '1NN_residuals' in cv_params and cv_params['1NN_residuals']:
                    # y_all = adjust_target_format(pd.concat([y_new,y_test_new]), target_type)
                    # y_new = y_all.loc[y_new.index]
                    # y_test_new = y_all.loc[y_test_new.index]
                    if target_type == 'binary':
                        y_test_new = (y_test_new==y_new.unique()[1]).astype(int)
                        y_new = (y_new==y_new.unique()[1]).astype(int)

                    from sklearn.neighbors import NearestNeighbors

                    nbrs = NearestNeighbors(n_neighbors=2).fit(X.select_dtypes(include=[np.number]))
                    distances, indices = nbrs.kneighbors(X.select_dtypes(include=[np.number]))
                    distances_test, indices_test = nbrs.kneighbors(X_test.select_dtypes(include=[np.number]))

                    y_lin = y.iloc[indices[:, 1]].values
                    # y_val_lin = lm.predict(X_val)
                    y_test_lin = y.iloc[indices_test[:, 1]].values
                    eval_metric_name = 'rmse'
                    problem_type = 'regression' 
                    init_params.update({'objective': 'regression'})
                else:
                    eval_metric_name = 'roc_auc' if target_type=='binary' else ('rmse' if target_type=='regression' else 'log_loss')
                    problem_type = target_type
                
                for prep in preprocessors:
                    X_new = prep.fit_transform(X_new, y_new)
                    X_test_new = prep.transform(X_test_new)

                if np.unique(X_new.columns).shape[0] != X_new.columns.shape[0]:
                    print("Warning: There are duplicate columns after preprocessing!")
                    # raise ValueError("There are duplicate columns after preprocessing!")
                    X_new = X_new.loc[:,~X_new.columns.duplicated()]
                    X_test_new = X_test_new.loc[:,~X_test_new.columns.duplicated()]

                model_key = model_name
                ag_bag = AGSingleBagWrapper(model_key, 
                                            init_params, 
                                            problem_type=problem_type, 
                                            eval_metric=eval_metric_name, 
                                            fit_kwargs=fit_kwargs,
                                            )
                
                if ('linear_residuals' in cv_params and cv_params['linear_residuals']) or ('1NN_residuals' in cv_params and cv_params['1NN_residuals']):
                    # y_use = y_new.copy().astype(float)
                    # y_use[y_use==1] = y_use[y_use==1]-y_lin[y_use==1]
                    # y_use[y_use==0] = y_use[y_use==0]+y_lin[y_use==0]
                    ag_bag.fit(X_new,y_new-y_lin)
                else:
                    ag_bag.fit(X_new,y_new)
                ag_meta = ag_bag.get_metadata()

                # y_new = (y_new==ag_bag.predictor.positive_class).astype(int)
                # y_test_new = (y_test_new==ag_bag.predictor.positive_class).astype(int)

                if problem_type=='binary':
                    y_test_pred = ag_bag.predict_proba(X_test_new)[ag_bag.predictor.positive_class]
                elif problem_type=='multiclass':
                    y_test_pred = ag_bag.predict_proba(X_test_new)
                else:
                    y_test_pred = ag_bag.predict(X_test_new)
                y_test_pred_per_val = ag_bag.bag_artifact(X_test_new)['pred_proba_test_per_child']

                if ('linear_residuals' in cv_params and cv_params['linear_residuals']) or ('1NN_residuals' in cv_params and cv_params['1NN_residuals']):
                    y_test_pred += y_test_lin
                    y_test_pred_per_val = [y_pred_i + y_test_lin for y_pred_i in y_test_pred_per_val]

                    if target_type=='binary':
                        y_test_pred = np.clip(y_test_pred, 0., 1.)
                        y_test_pred_per_val = [np.clip(y_pred_i, 0., 1.) for y_pred_i in y_test_pred_per_val]
                        eval_metric = roc_auc_score
                    elif target_type=='regression':
                        eval_metric = lambda y_true, y_pred: -root_mean_squared_error(y_true, y_pred)
                    elif target_type=='multiclass':
                        raise NotImplementedError()
                    # y_val_pred = pd.concat([pd.Series(pred[ag_bag.get_per_child_val_idx()[num]],index=ag_bag.get_per_child_val_idx()[num]) for num, pred in enumerate(ag_bag.get_per_child_test(X))]).sort_index()
                    # TODO: Fix bug when applying those functions with linear residuals and arithmetic interactions
                    val_preds = [pred[ag_bag.get_per_child_val_idx()[num]]+y_lin[ag_bag.get_per_child_val_idx()[num]] for num, pred in enumerate(ag_bag.get_per_child_test(X))]
                    val_preds = [np.clip(pred, 0., 1.) if target_type=='binary' else pred for pred in val_preds]
                    val_scores = [eval_metric(y_new.iloc[ag_bag.get_per_child_val_idx()[num]], pred) for num, pred in enumerate(val_preds)]
                    # mean_val_score = eval_metric(y_new,y_val_pred)
                    mean_val_score = np.mean(val_scores)
                else:
                    eval_metric = ag_bag.predictor.eval_metric
                    if problem_type in ['binary', 'multiclass']:
                        y_test_new = y_test_new.map(ag_bag.predictor.class_labels_internal_map)
                        y_new = y_new.map(ag_bag.predictor.class_labels_internal_map)

                    if "ag_args_ensemble" in init_params and 'refit_folds' in init_params["ag_args_ensemble"] and init_params["ag_args_ensemble"]["refit_folds"]:
                        mean_val_score = ag_meta['info']['val_score']
                        val_scores =  [ag_meta['info']['val_score']]*8
                        
                    else:
                        mean_val_score = np.mean([ag_meta['info']['children_info'][f'S1F{i}']['val_score'] for i in range(1,ag_meta['info']['bagged_info']['num_child_models']+1)])
                        val_scores =  [ag_meta['info']['children_info'][f'S1F{i}']['val_score'] for i in range(1,ag_meta['info']['bagged_info']['num_child_models']+1)]

                if post_predict_duplicate_mapping:
                    val_preds = [pred[ag_bag.get_per_child_val_idx()[num]] for num, pred in enumerate(ag_bag.get_per_child_test(X))]
                    X_str = X.astype(str).sum(axis=1)
                    dupe_map = dict(y_new.astype(float).groupby(X_str).mean())
                    cnt_map = dict(y_new.astype(float).groupby(X_str).count())
                    dupe_map = {k: v for k, v in dupe_map.items() if cnt_map[k] > 1}
                    dupe_min = dict(y_new.astype(float).groupby(X_str).min())
                    dupe_max = dict(y_new.astype(float).groupby(X_str).max())
                    dupe_map = {k: v for k, v in dupe_map.items() if dupe_min[k]==dupe_max[k]}

                    X_test_str = X_test.astype(str).sum(axis=1)
                    any_dupes = X_test_str.apply(lambda x: x in dupe_map).sum() > 0
                    if any_dupes:
                        # if target_type == 'regression':
                        new_pred = [float(dupe_map[i]) if i in dupe_map else float(y_test_pred.iloc[num])  for num, i in enumerate(X_test_str)]
                        y_test_pred = pd.Series(new_pred, index=y_test_pred.index, name=y_test_pred.name)

                        y_test_pred_per_val = [pd.Series([float(dupe_map[i]) if i in dupe_map else float(pred[num])  for num, i in enumerate(X_test_str)], index=y_test_pred.index, name=y_test_pred.name) for pred in y_test_pred_per_val]

                        val_preds = [[float(dupe_map[i]) if i in dupe_map else float(pred[num_i]) for num_i, i in enumerate(X_str.iloc[ag_bag.get_per_child_val_idx()[num]])] for num, pred in enumerate(val_preds)]
                        val_scores =  [eval_metric(y_new.iloc[ag_bag.get_per_child_val_idx()[num]], val_preds[num]) for num, i in enumerate(val_preds)]
                        mean_val_score = np.mean(val_scores)


                        
                        # elif target_type == 'binary':
                        #     new_pred_1 = np.array([dupe_map[i] if i in dupe_map else float(y_test_pred.iloc[num, 1]) for num, i in enumerate(X_test_str)])
                        #     new_pred_0 = np.array([dupe_map[i] if i in dupe_map else float(y_test_pred.iloc[num, 0]) for num, i in enumerate(X_test_str)])
                        #     auc_1 = [roc_auc_score(new_pred_1.round(), y_pred_adapted.iloc[:,i].values) for i in [0,1]]
                        #     auc_0 = [roc_auc_score(new_pred_0.round(), y_pred_adapted.iloc[:,i].values) for i in [0,1]]

                        #     # FIXME: Get rid of the hack to infer which colum matches the preprocessing vs what AG does
                        #     if max(auc_1) > max(auc_0):
                        #         new_pred = new_pred_1
                        #     else:
                        #         new_pred = new_pred_0

                        #     if new_pred[0] > new_pred[1]:
                        #         use_col = 0
                        #         other_col = 1
                        #     else:
                        #         use_col = 1
                        #         other_col = 0

                        #     y_pred_adapted.iloc[:,use_col] = new_pred
                        #     y_pred_adapted.iloc[:,other_col] = 1 - y_pred_adapted.iloc[:,use_col]


                dat_results = {
                    'init_params': init_params,
                    'preprocessors': [i.__name__ for i in preprocessors],
                    'val_scores': val_scores,
                    'mean_val_score': mean_val_score,
                    # TODO: Align logic of my CV and this part
                    'test_preds_per_val': y_test_pred_per_val,
                    'test_preds': y_test_pred,
                    'feature_importances': None
                }

                if model_name == 'GBM':    
                    dat_results['iterations'] = [ag_meta['info']['children_info'][f'S1F{i}']['hyperparameters_fit']['num_boost_round'] for i in range(1,ag_meta['info']['bagged_info']['num_child_models']+1)]

                # if target_type=='binary':
                #     metric = roc_auc_score
                # elif target_type=='regression':
                #     metric = lambda y_true, y_pred: -root_mean_squared_error(y_true, y_pred)
                # elif target_type=='multiclass':
                #     metric = lambda y_true, y_pred: -log_loss(y_true, y_pred)

                dat_results['bagged_test_score'] = eval_metric(y_test_new, y_test_pred)
                dat_results['test_scores'] = [eval_metric(y_test_new, y_pred_i) for y_pred_i in y_test_pred_per_val]                    
            
            return dat_results

def run(
        base_dir: str,
        dataset_names: list,
        considered_presets: list = None,
        rerun_presets: list = None,
        use_test='TabArena',
        cv_type: Literal['custom', 'tabarena'] = 'custom',
        experimental_presets=False,
        model_name='GBM',
        ):
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if rerun_presets is None:
        rerun_presets = []

    results = {}
    for dataset_name in dataset_names: #os.listdir(base_dir):
        print(dataset_name)
        if use_test is None:
            X, y, target_type = get_dataset(dataset_name)
            X_test, y_test = None, None
        elif use_test=='TabArena-Lite':
            metadata = load_task_metadata()
            tid = int(metadata.loc[metadata.name==dataset_name,'tid'].iloc[0])
            task = OpenMLTaskWrapper(openml.tasks.get_task(tid))
            X, y, X_test, y_test = task.get_train_test_split(fold=0, repeat=0)
            target_type = task.problem_type
        
        if os.path.exists(f"{base_dir}/{dataset_name}_preset_results.pkl"):
            with open(f"{base_dir}/{dataset_name}_preset_results.pkl", "rb") as f:
                dat_results = pickle.load(f)
                available_presets = list(dat_results.keys())

            print(f"Results for dataset {dataset_name} already exists with presets: {list(dat_results.keys())}.")
            # if use_test is None:
            #     print(f"Results for dataset {dataset_name} already exists with presets: {pd.Series({preset: res['avg_score'] for preset, res in dat_results.items()}).sort_values()}.")
            # else:
            #     print(f"Results for dataset {dataset_name} already exists with presets: {pd.DataFrame({preset: [res['avg_score'], res['avg_test_score']] for preset, res in dat_results.items()},index=['Val', 'Test']).T.sort_values('Test')}.")

        else:
            print("Running presets from scratch")
            dat_results = {}

        presets = LGBPresetRegistry(target_type, use_experimental=experimental_presets)

        if considered_presets is None:
            all_presets = presets.available_presets()
        else:
            all_presets = [preset for preset in considered_presets if preset in presets.available_presets()]

        all_presets = filter_presets(all_presets, target_type, X, y)

        print(f'Run the following presets: {[i for i in all_presets if i not in dat_results or i in rerun_presets]}')

        for preset in all_presets:
            if preset in dat_results and preset not in rerun_presets:
                continue
            print('Train preset', preset)
            # try:
            init_params, preprocessors, cv_params = presets.get_params(target_type, preset=preset)
            dat_results[preset] = run_preset(X=X, y=y, X_test=X_test, y_test=y_test, 
                                            target_type=target_type, init_params=init_params, 
                                            preprocessors=preprocessors, cv_params=cv_params, 
                                            cv_type=cv_type,
                                            model_name=model_name
                                            )

            print(f"{preset}: {np.mean(dat_results[preset]['val_scores'])}", {np.mean(dat_results[preset]['test_scores']) if 'test_scores' in dat_results[preset] else None}, {dat_results[preset]['bagged_test_score'] if 'bagged_test_score' in dat_results[preset] else None})

            with open(f"{base_dir}/{dataset_name}_preset_results.pkl", "wb") as f:
                pickle.dump(dat_results, f)
            # except Exception as e:
            #     print(f"Preset {preset} failed for dataset {dataset_name} due to {e}.")
            #     continue
        
        if cv_type == 'tabarena':
            print(f"Results for dataset {dataset_name}: {pd.DataFrame({preset: [res['mean_val_score'], res['bagged_test_score']] for preset, res in dat_results.items()},index=['Val', 'Test']).T.sort_values('Test').tail(50)}.")

        else:
            if use_test is None:
                print(f"Results for dataset {dataset_name}: {pd.Series({preset: res['avg_score'] for preset, res in dat_results.items()}).sort_values()}.")
            else:
                print(f"Results for dataset {dataset_name}: {pd.DataFrame({preset: [res['avg_score'], res['avg_test_score']] for preset, res in dat_results.items()},index=['Val', 'Test']).T.sort_values('Test')}.")
        
        results[dataset_name] = dat_results
    
    return results


def load_results(base_dir: str = '/ceph/atschalz/auto_prep/preset_results'):
    from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
    metadata = load_task_metadata()
    dataset_names = metadata.sort_values('n_samples_train_per_fold').name.tolist()

    results = {}
    for dataset_name in dataset_names: #os.listdir(base_dir):
        if os.path.exists(f"{base_dir}/{dataset_name}_preset_results.pkl"):
            with open(f"{base_dir}/{dataset_name}_preset_results.pkl", "rb") as f:
                dat_results = pickle.load(f)
        else:
            print(f"Results for dataset {dataset_name} do not exist.")
            continue
        results[dataset_name] = dat_results

    return results

if __name__ == "__main__":
    from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
    metadata = load_task_metadata()
    dataset_names = metadata.sort_values('n_samples_train_per_fold').name.tolist()

    base_dir = '/ceph/atschalz/auto_prep/preset_results/bagging_tabarena_lite'

    results = run(base_dir, use_test='TabArena-Lite', cv_type = 'tabarena',
        # considered_presets=[""],
        # rerun_presets=[''],
        dataset_names=dataset_names,
        experimental_presets=False, # Set to True to include experimental presets. Define new presets in tabprep/presets/lgb_presets_experimental.py in the same way as done in tabprep/presets/lgb_presets.py
        model_name='GBM'

    )

