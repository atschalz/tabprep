import openml
import pandas as pd
import numpy as np
from tabrepo.nips2025_utils import tabarena_context
from tabprep.utils import get_benchmark_dataIDs
from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
import os
import matplotlib.pyplot as plt

def get_val_test_corr(tids, model_names=['LightGBM', 'CatBoost', 'XGBoost', 'TabPFNv2_GPU', 'RealMLP', 'TabM_GPU']):
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
                        'c1', os.path.exists(f"results/{exp_name}/data/{model_name}_c1_BAG_L1/{tid}/{repeat}_{fold}/results.pkl")
                    ])
                    for r in range(1,200):
                        path = f"results/{exp_name}/data/{model_name}_r{r}_BAG_L1/{tid}/{repeat}_{fold}/results.pkl"
                        availabilities.append([
                           tid, repeat, fold, model_name,
                           f'r{r}', os.path.exists(path)
                        ])

    return pd.DataFrame(availabilities, columns=[
        'task_id', 'repeat', 'fold', 'model_name', 'config_type', 'available'
    ])


if __name__ == "__main__":
    tids, dids = get_benchmark_dataIDs('TabArena')
    availabilities = check_result_availability('all_in_one', ['LightGBM'], tids, list(range(3)), list(range(3)))
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