import openml
import pandas as pd
import numpy as np
from tabprep.utils import get_benchmark_dataIDs
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
import os
import matplotlib.pyplot as plt

def get_val_test_corr(tids, model_names=['LightGBM', 'CatBoost', 'XGBoost', 'TabPFNv2_GPU', 'RealMLP', 'TabM_GPU']):
    tabarena_context = TabArenaContext()
    res_mean = pd.DataFrame(columns=model_names)
    res_std = pd.DataFrame(columns=model_names)
    res_all = {}
    for tid, did in zip(tids, dids):
        res_all[tid] = {}
        task = openml.tasks.get_task(tid)  # to check if the datasets are available
        data = openml.datasets.get_dataset(did)  # to check if the datasets are available
        for model_name in model_names:
            res_all[tid][model_name] = {}
            res = tabarena_context.load_config_results(model_name)
            res = res.loc[res.dataset==data.name]
            cor_lst = []
            for fold in res.fold.unique():
                res_use = res.loc[res.fold==fold].copy()
                cor_lst.append(res_use[['metric_error','metric_error_val']].corr('spearman').iloc[0,1])

            res_mean.loc[data.name, model_name] = np.mean(cor_lst)
            res_std.loc[data.name, model_name] = np.std(cor_lst)
            res_all[tid][model_name][data.name] = cor_lst
    return res_mean, res_std, res_all


def check_result_availability(exp_name, model_names, tids, repeats, folds):
    availabilities = []
    for tid in tids:
        for repeat in repeats:
            for fold in folds:
                for model_name in model_names:
                    availabilities.append([
                        tid, repeat, fold, model_name,
                        'c1', os.path.exists(f"../results/{exp_name}/data/{model_name}_c1_BAG_L1/{tid}/{repeat}_{fold}/results.pkl")
                    ])
                    for r in range(1,200):
                        path = f"../results/{exp_name}/data/{model_name}_r{r}_BAG_L1/{tid}/{repeat}_{fold}/results.pkl"
                        availabilities.append([
                           tid, repeat, fold, model_name,
                           f'r{r}', os.path.exists(path)
                        ])

    return pd.DataFrame(availabilities, columns=[
        'task_id', 'repeat', 'fold', 'model_name', 'config_type', 'available'
    ])

def load_all_configs(exp_name, repeats=3, folds=3, trials = 200, use_datasets=None, impute = True):
    tabarena_context = TabArenaContext()

    # exp_name = 'all_in_one'
    # repeats = 3
    # use_datasets = ['hiva_agnostic', 'Bioresponse', 'kddcup09_appetency', 'MIC','Diabetes130US', 'anneal', 'qsar-biodeg']
    # trials = 200

    tids, dids = get_benchmark_dataIDs('TabArena')
    ta_results = tabarena_context.load_config_results('LightGBM')
    new_entries = []
    results_lst = []
    for tid, did in zip(tids, dids):
        task = openml.tasks.get_task(tid)  # to check if the datasets are available
        data = openml.datasets.get_dataset(did)  # to check if the datasets are available
        if use_datasets is not None and data.name not in use_datasets:
            continue
        for repeat in range(repeats):
            for fold in range(folds):
                new_res_found = False
                try:
                    path = f"../results/{exp_name}/data/LightGBM_c1_BAG_L1/{tid}/{repeat}_{fold}/results.pkl"
                    with open(path, "rb") as f:
                        results = pd.read_pickle(f)
                        new_entry = pd.Series({
                            'dataset': results['task_metadata']['name'],
                            'fold': results['task_metadata']['fold']+(3*results['task_metadata']['repeat']),
                            'method': results['framework'],
                            'metric_error': float(results['metric_error']),
                            'time_train_s': float(results['time_train_s']),
                            'time_infer_s': float(results['time_infer_s']),
                            'metric': results['metric'],
                            'problem_type': results['problem_type'],
                            'metric_error_val': float(results['metric_error_val']),
                            'method_type': 'config',
                            'config_type': results['framework'].split('_')[0]+'(CSV)',
                            'ta_name': 'CSV',
                            'ta_suite': 'CSV',
                            'imputed': False
                        })
                        new_entries.append(new_entry)
                        results['framework'] += 'CSV'
                        results_lst.append(results)
                        new_res_found = True

                    for r in range(1,trials+1):
                        path = f"../results/{exp_name}/data/LightGBM_r{r}_BAG_L1/{tid}/{repeat}_{fold}/results.pkl"
                        with open(path, "rb") as f:
                            results = pd.read_pickle(f)
                            new_entry = pd.Series({
                                'dataset': results['task_metadata']['name'],
                                'fold': results['task_metadata']['fold']+(3*results['task_metadata']['repeat']),
                                'method': results['framework'],
                                'metric_error': results['metric_error'],
                                'time_train_s': results['time_train_s'],
                                'time_infer_s': results['time_infer_s'],
                                'metric': results['metric'],
                                'problem_type': results['problem_type'],
                                'metric_error_val': results['metric_error_val'],
                                'method_type': 'config',
                                'config_type': results['framework'].split('_')[0]+'(CSV)',
                                'ta_name': 'CSV',
                                'ta_suite': 'CSV',
                                'imputed': False
                            })
                            new_entries.append(new_entry)
                            results['framework'] += 'CSV'
                            results_lst.append(results)
                            new_res_found = True

                except FileNotFoundError:
                    print(f"Results for {data.name} at {path} not found. Fill with TabArena results.")
                    # res = ta_results.loc[ta_results.dataset==data.name]
                    # res.loc[res.groupby('fold')['metric_error_val'].idxmin(), 'metric_error'].mean()
                if not new_res_found and impute:
                    rep_fold = fold+(3*repeat)
                    res = ta_results.loc[ta_results.dataset==data.name]
                    res = res.loc[res['fold']==rep_fold]
                    new_entry = res[['dataset', 'fold', 'method', 'metric_error', 'time_train_s',
                        'time_infer_s', 'metric', 'problem_type', 'metric_error_val',
                        'method_type', 'config_type', 'ta_name', 'ta_suite']]
                    new_entry.loc[:, 'config_type'] = new_entry['method'].apply(lambda x: x.split('_')[0] + '(CSV)')
                    new_entry.loc[:, 'ta_name'] = 'CSV'
                    new_entry.loc[:, 'ta_suite'] = 'CSV'
                    new_entry.loc[:, 'imputed'] = True
                    new_entries += [new_entry.loc[i] for i in new_entry.index]
    model_results = pd.concat(new_entries, ignore_index=True,axis=1).T
    return model_results


def compare_new_to_old(model_results):
    tabarena_context = TabArenaContext()
    ta_results = tabarena_context.load_config_results('LightGBM')
    improved_trials = {}
    improved_trials_val = {}
    for dataset_name in model_results.dataset.unique():
        my = model_results.loc[model_results.dataset==dataset_name]
        ta = ta_results.loc[ta_results.dataset==dataset_name]
        merged = pd.merge(my, ta, on=['dataset', 'fold', 'method'], suffixes=('_my', '_ta'))
        merged[['fold','method', 'metric_error_my', 'metric_error_ta']]
        improved_trials[dataset_name] = sum(np.sign(merged[['metric_error_my', 'metric_error_ta']].diff(axis=1).iloc[:,1]))/merged.shape[0]
        improved_trials_val[dataset_name] = sum(np.sign(merged[['metric_error_val_my', 'metric_error_val_ta']].diff(axis=1).iloc[:,1]))/merged.shape[0]    # print(f"{dataset_name}: {improved_trials[dataset_name]:.2f}")
    df_improved_trials = pd.concat([pd.Series(improved_trials),pd.Series(improved_trials_val)],axis=1)
    df_improved_trials.columns = ['Test', 'Val']
    

    return df_improved_trials.sort_values('Test', ascending=False)


def compare_to_others(model_results, lgb_use = 'tuned'):
    from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
    tabarena_context = TabArenaContext()

    # exp_name = 'all_in_one_0806_v2'
    # model_results = pd.read_csv(f"../model_results_{exp_name}.csv")

    default = model_results.loc[model_results['method'].apply(lambda x: 'c1' in x)]
    default['method'] = 'LightGBM(CSV)'
    default['config_type'] = 'default'
    tuned = model_results.loc[model_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error_val'].idxmin()).values]
    tuned['method'] = 'LightGBM(CSV)'
    tuned['config_type'] = 'tuned'
    tuned_ensemble = model_results.loc[model_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error_val'].idxmin()).values]
    tuned_ensemble['method'] = 'LightGBM(CSV)'
    tuned_ensemble['config_type'] = 'tuned + ensemble'

    ta_results = tabarena_context.load_config_results('LightGBM')
    ta_results = ta_results[ta_results.fold<=8]
    ta_default = ta_results.loc[ta_results['method'].apply(lambda x: 'c1' in x)]
    ta_default['method'] = 'LightGBM'
    ta_default['config_type'] = 'default'
    ta_tuned = ta_results.loc[ta_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error_val'].idxmin()).values]
    ta_tuned['method'] = 'LightGBM'
    ta_tuned['config_type'] = 'tuned'
    ta_tuned_ensemble = ta_results.loc[ta_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error_val'].idxmin()).values]
    ta_tuned_ensemble['method'] = 'LightGBM'
    ta_tuned_ensemble['config_type'] = 'tuned + ensemble'

    ag_results = tabarena_context.load_config_results('AutoGluon_v130')
    ag_results = ag_results[ag_results.fold<=8]

    tabpfn_results = tabarena_context.load_config_results('TabPFNv2_GPU')
    tabpfn_results = tabpfn_results[tabpfn_results.fold<=8]
    tabpfn_default = tabpfn_results.loc[tabpfn_results['method'].apply(lambda x: 'c1' in x)]
    tabpfn_default['method'] = 'TabPFNv2'
    tabpfn_default['config_type'] = 'default'
    tabpfn_tuned = tabpfn_results.loc[tabpfn_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error_val'].idxmin()).values]
    tabpfn_tuned['method'] = 'TabPFNv2'
    tabpfn_tuned['config_type'] = 'tuned'

    rf_results = tabarena_context.load_config_results('RandomForest')
    rf_results = rf_results[rf_results.fold<=8]
    rf_default = rf_results.loc[rf_results['method'].apply(lambda x: 'c1' in x)]
    rf_default['method'] = 'RF'
    rf_default['config_type'] = 'default'

    if lgb_use == 'default':
        lgb_use = default
        lgb_ta_use = ta_default
    elif lgb_use == 'tuned':
        lgb_use = tuned
        lgb_ta_use = ta_tuned

    comp_df = pd.concat([
        rf_default.groupby('dataset')['metric_error'].mean(),
        tabpfn_default.groupby('dataset')['metric_error'].mean(),
        tabpfn_tuned.groupby('dataset')['metric_error'].mean(),
        ag_results.groupby('dataset')['metric_error'].mean(),
        lgb_ta_use.groupby('dataset')['metric_error'].mean(),
        lgb_use.groupby('dataset')['metric_error'].mean()],axis=1)
        # ta_default.groupby('dataset')['metric_error'].mean(),
        # default.groupby('dataset')['metric_error'].mean()
    comp_df.columns = ['RF (default)', 'TabPFNv2 (default)', 'TabPFNv2 (tuned)', 'AutoGluon', 'TabArena', 'New']
    return comp_df

if __name__ == "__main__":
    tids, dids = get_benchmark_dataIDs('TabArena')
    exp_name = 'all_in_one_0812'
    availabilities = check_result_availability(exp_name, ['LightGBM'], tids, list(range(3)), list(range(3)))
    print(availabilities['available'].mean())
    model_results = load_all_configs(exp_name, repeats=3, folds=3, trials = 200, impute=False)
    df_improved_trials = compare_new_to_old(model_results)
    print(df_improved_trials)
    print(model_results.groupby('dataset')['metric_error'].count().sum()/(1809*51))
    
    # How many results are finished?
    print(model_results.groupby('dataset')['metric_error'].count())
    
    comp_df_others = compare_to_others(model_results)

    tabarena_context = TabArenaContext()
    ta_results = tabarena_context.load_config_results('LightGBM')
    ta_results = ta_results.loc[ta_results['fold']<9]
    comp_df = pd.merge(model_results,ta_results,on=['dataset', 'fold','method'])
    print((comp_df['metric_error_x']-comp_df['metric_error_y']).groupby(comp_df['dataset']).mean())
    # print(pd.concat([comp_df.groupby('dataset')['metric_error_x'].min(),comp_df.groupby('dataset')['metric_error_y'].min()], axis=1))    
    print(pd.concat([comp_df.groupby('dataset').apply(lambda x: x['metric_error_x'][x['metric_error_val_x'].idxmin()]),comp_df.groupby('dataset').apply(lambda x: x['metric_error_y'][x['metric_error_val_y'].idxmin()])], axis=1))    
    
    ### Determine the targeted time
    target_time = 120
    print(dict((target_time/(model_results.groupby('dataset')['time_train_s'].max()/60)).astype(int)))

    # Compare multipe folds of one dataset
    dat = 'jm1'
    pd.concat([
        model_results.loc[np.logical_and(model_results.dataset==dat, model_results.method=='LightGBM_c1_BAG_L1'),'metric_error'].reset_index(drop=True),
        ta_results.loc[np.logical_and(ta_results.dataset==dat, ta_results.method=='LightGBM_c1_BAG_L1'),'metric_error'].reset_index(drop=True),
    ], axis=1, ignore_index=True)
    
    # Test for validation overfitting
    mean_val_test_corr = pd.DataFrame(ta_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error'].corr(x['metric_error_val']))).unstack().mean(axis=1).sort_values()
    std_val_test_corr = pd.DataFrame(ta_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error'].corr(x['metric_error_val']))).unstack().std(axis=1)
    std_val_test_corr = std_val_test_corr.loc[mean_val_test_corr.index]
    print_val_test_corr = mean_val_test_corr.round(2).astype(str) + ' (' + std_val_test_corr.round(2).astype(str) + ')'

    new_mean_val_test_corr = pd.DataFrame(model_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error'].corr(x['metric_error_val']))).unstack().mean(axis=1).sort_values()
    new_std_val_test_corr = pd.DataFrame(model_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error'].corr(x['metric_error_val']))).unstack().std(axis=1)
    new_std_val_test_corr = new_std_val_test_corr.loc[new_mean_val_test_corr.index]
    new_print_val_test_corr = new_mean_val_test_corr.round(2).astype(str) + ' (' + new_std_val_test_corr.round(2).astype(str) + ')'
    print(pd.concat([print_val_test_corr,new_print_val_test_corr], axis=1))
    
    # model_results_filtered = model_results.loc[np.logical_not(model_results['imputed'].values)]
    # available = model_results_filtered.method.unique()
    # ta_sub = ta_results.loc[ta_results.method.apply(lambda x: x in available)]
    # my_sub = model_results.loc[model_results.method.apply(lambda x: x in available)]
    # ta_sub = ta_sub[ta_sub.fold<=8]
    # comp_sub = pd.concat([my_sub.groupby('dataset')['metric_error'].min(),ta_sub.groupby('dataset')['metric_error'].min()], axis=1)
    # print(comp_sub.loc[comp_sub.diff(axis=1).iloc[:,1]!=0])
    
    model_results.to_csv(f"model_results_{exp_name}.csv", index=False)
    print('DONE')

# tabarena_context = TabArenaContext()

# tids, dids = get_benchmark_dataIDs('TabArena')
# ta_results = tabarena_context.load_config_results('LightGBM')
# new_entries = []
# results_lst = []
# for tid, did in zip(tids, dids):
#     task = openml.tasks.get_task(tid)  # to check if the datasets are available
#     data = openml.datasets.get_dataset(did)  # to check if the datasets are available
#     for repeat in range(3):
#         for fold in range(3):
#             new_res_found = False
#             try:
#                 with open(f"../results/{exp_name}/data/LightGBM_c1_BAG_L1/{tid}/{repeat}_{fold}/results.pkl", "rb") as f:
#                     results = pd.read_pickle(f)
#                     new_entry = pd.Series({
#                         'dataset': results['task_metadata']['name'],
#                         'fold': results['task_metadata']['fold']+(3*results['task_metadata']['repeat']),
#                         'method': results['framework'],
#                         'metric_error': float(results['metric_error']),
#                         'time_train_s': float(results['time_train_s']),
#                         'time_infer_s': float(results['time_infer_s']),
#                         'metric': results['metric'],
#                         'problem_type': results['problem_type'],
#                         'metric_error_val': float(results['metric_error_val']),
#                         'method_type': 'config',
#                         'config_type': results['framework'].split('_')[0]+'(CSV)',
#                         'ta_name': 'CSV',
#                         'ta_suite': 'CSV',
#                     })
#                     new_entries.append(new_entry)
#                     results['framework'] += 'CSV'
#                     results_lst.append(results)
#                     new_res_found = True

#                 for r in range(1,200):
#                     path = f"../results/{exp_name}/data/LightGBM_r{r}_BAG_L1/{tid}/{repeat}_{fold}/results.pkl"
#                     with open(path, "rb") as f:
#                         results = pd.read_pickle(f)
#                         new_entry = pd.Series({
#                             'dataset': results['task_metadata']['name'],
#                             'fold': results['task_metadata']['fold']+(3*results['task_metadata']['repeat']),
#                             'method': results['framework'],
#                             'metric_error': results['metric_error'],
#                             'time_train_s': results['time_train_s'],
#                             'time_infer_s': results['time_infer_s'],
#                             'metric': results['metric'],
#                             'problem_type': results['problem_type'],
#                             'metric_error_val': results['metric_error_val'],
#                             'method_type': 'config',
#                             'config_type': results['framework'].split('_')[0]+'(CSV)',
#                             'ta_name': 'CSV',
#                             'ta_suite': 'CSV',
#                         })
#                         new_entries.append(new_entry)
#                         results['framework'] += 'CSV'
#                         results_lst.append(results)
#                         new_res_found = True

#             except FileNotFoundError:
#                 print(f"Results for {data.name} at {path} not found. Fill with TabArena results.")
#                 # res = ta_results.loc[ta_results.dataset==data.name]
#                 # res.loc[res.groupby('fold')['metric_error_val'].idxmin(), 'metric_error'].mean()
#             if not new_res_found:
#                 rep_fold = fold+(3*repeat)
#                 res = ta_results.loc[ta_results.dataset==data.name]
#                 res = res.loc[res['fold']==rep_fold]
#                 new_entry = res[['dataset', 'fold', 'method', 'metric_error', 'time_train_s',
#                     'time_infer_s', 'metric', 'problem_type', 'metric_error_val',
#                     'method_type', 'config_type', 'ta_name', 'ta_suite']]
#                 new_entry.loc[:, 'config_type'] = new_entry['method'].apply(lambda x: x.split('_')[0] + '(CSV)')
#                 new_entry.loc[:, 'ta_name'] = 'CSV'
#                 new_entry.loc[:, 'ta_suite'] = 'CSV'
#                 new_entries += [new_entry.loc[i] for i in new_entry.index]
# model_results = pd.concat(new_entries, ignore_index=True,axis=1).T
# model_results


# if __name__ == "__main__":
#     exp_name = 'csv'
    
#     import matplotlib.pyplot as plt
#     import openml
#     from utils import get_benchmark_dataIDs
#     from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
#     download_results = "auto"  # results must be downloaded for the script to work
#     elo_bootstrap_rounds = 100  # 1 for toy, 100 for paper
#     save_path = "output_paper_results"  # folder to save all figures and tables
#     use_latex = False  # Set to True if you have the appropriate latex packages installed for nicer figure style

#     tabarena_context = TabArenaContext()

#     tids, dids = get_benchmark_dataIDs('TabArena')
#     ta_results = tabarena_context.load_config_results('LightGBM')
#     new_entries = []
#     results_lst = []
#     for tid, did in zip(tids, dids):
#         task = openml.tasks.get_task(tid)  # to check if the datasets are available
#         data = openml.datasets.get_dataset(did)  # to check if the datasets are available
#         for repeat in range(3):
#             for fold in range(3):
#                 new_res_found = False
#                 try:
#                     with open(f"results/{exp_name}/data/LightGBM_c1_BAG_L1/{tid}/{repeat}_{fold}/results.pkl", "rb") as f:
#                         results = pd.read_pickle(f)
#                         new_entry = pd.Series({
#                             'dataset': results['task_metadata']['name'],
#                             'fold': results['task_metadata']['fold']+(3*results['task_metadata']['repeat']),
#                             'method': results['framework'],
#                             'metric_error': float(results['metric_error']),
#                             'time_train_s': float(results['time_train_s']),
#                             'time_infer_s': float(results['time_infer_s']),
#                             'metric': results['metric'],
#                             'problem_type': results['problem_type'],
#                             'metric_error_val': float(results['metric_error_val']),
#                             'method_type': 'config',
#                             'config_type': results['framework'].split('_')[0]+'(CSV)',
#                             'ta_name': 'CSV',
#                             'ta_suite': 'CSV',
#                         })
#                         new_entries.append(new_entry)
#                         results['framework'] += 'CSV'
#                         results_lst.append(results)
#                         new_res_found = True

#                     for r in range(1,199):
#                         path = f"results/{exp_name}/data/LightGBM_r{r}_BAG_L1/{tid}/{repeat}_{fold}/results.pkl"
#                         with open(path, "rb") as f:
#                             results = pd.read_pickle(f)
#                             new_entry = pd.Series({
#                                 'dataset': results['task_metadata']['name'],
#                                 'fold': results['task_metadata']['fold']+(3*results['task_metadata']['repeat']),
#                                 'method': results['framework'],
#                                 'metric_error': results['metric_error'],
#                                 'time_train_s': results['time_train_s'],
#                                 'time_infer_s': results['time_infer_s'],
#                                 'metric': results['metric'],
#                                 'problem_type': results['problem_type'],
#                                 'metric_error_val': results['metric_error_val'],
#                                 'method_type': 'config',
#                                 'config_type': results['framework'].split('_')[0]+'(CSV)',
#                                 'ta_name': 'CSV',
#                                 'ta_suite': 'CSV',
#                             })
#                             new_entries.append(new_entry)
#                             results['framework'] += 'CSV'
#                             results_lst.append(results)
#                             new_res_found = True

#                 except FileNotFoundError:
#                     print(f"Results for {data.name} at {path} not found. Fill with TabArena results.")
#                     # res = ta_results.loc[ta_results.dataset==data.name]
#                     # res.loc[res.groupby('fold')['metric_error_val'].idxmin(), 'metric_error'].mean()
#                 if not new_res_found:
#                     rep_fold = fold+(3*repeat)
#                     res = ta_results.loc[ta_results.dataset==data.name]
#                     res = res.loc[res['fold']==rep_fold]
#                     new_entry = res[['dataset', 'fold', 'method', 'metric_error', 'time_train_s',
#                         'time_infer_s', 'metric', 'problem_type', 'metric_error_val',
#                         'method_type', 'config_type', 'ta_name', 'ta_suite']]
#                     new_entry.loc[:, 'config_type'] = new_entry['method'].apply(lambda x: x.split('_')[0] + '(CSV)')
#                     new_entry.loc[:, 'ta_name'] = 'CSV'
#                     new_entry.loc[:, 'ta_suite'] = 'CSV'
#                     new_entries += [new_entry.loc[i] for i in new_entry.index]
#     model_results = pd.concat(new_entries, ignore_index=True,axis=1).T

#     default = model_results.loc[model_results['method'].apply(lambda x: 'c1' in x)]
#     default.loc[:, 'method'] = 'LightGBM(CSV)'
#     default.loc[:, 'config_type'] = 'default'
#     tuned = model_results.loc[model_results.groupby(['dataset', 'fold']).apply(lambda x: x['metric_error_val'].idxmin(), include_groups=False).values]
#     tuned.loc[:, 'method'] = 'LightGBM(CSV)'
#     tuned.loc[:, 'config_type'] = 'tuned'

#     ta_res_all = tabarena_context.load_results_paper(['LightGBM'])
#     ta_res_def = ta_res_all.loc[ta_res_all.method=='GBM (default)']
#     ta_res_tuned = ta_res_all.loc[ta_res_all.method=='GBM (tuned)']
#     ta_res_tunedens = ta_res_all.loc[ta_res_all.method=='GBM (tuned + ensemble)']

#     default_comparison = pd.concat(
#         [default.groupby('dataset')['metric_error'].mean(), 
#          ta_res_def.groupby('dataset')['metric_error'].mean()],
#         axis=1, keys=['default', 'ta_default']
#     )

#     plt.figure(figsize=(8, 5))
#     plt.figure(figsize=(8, 5))
#     plt.bar(default_comparison.index.astype(str), default_comparison.diff(axis=1).iloc[:,1]/default_comparison['ta_default'])
#     plt.xlabel('Row index')
#     plt.ylabel('Difference (col1 − col2)')
#     plt.title('Per‑Row Difference Between col1 and col2')
#     plt.xticks(rotation=90)  # rotate indices if needed
#     plt.tight_layout()
#     plt.savefig("comparison_def.png")

#     tuned_comparison = pd.concat(
#         [tuned.groupby('dataset')['metric_error'].mean(), 
#          ta_res_tuned.groupby('dataset')['metric_error'].mean()],
#         axis=1, keys=['tuned', 'ta_tuned']
#     )
#     plt.figure(figsize=(8, 5))
#     plt.bar(tuned_comparison.index.astype(str), tuned_comparison.diff(axis=1).iloc[:,1]/tuned_comparison['ta_tuned'])
#     plt.xlabel('Row index')
#     plt.ylabel('Difference (col1 − col2)')
#     plt.title('Per‑Row Difference Between col1 and col2')
#     plt.xticks(rotation=90)  # rotate indices if needed
#     plt.tight_layout()
#     plt.savefig("comparison_tuned.png")



#     print("!")