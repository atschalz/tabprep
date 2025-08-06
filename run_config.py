from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from tabrepo import EvaluationRepository, Evaluator
from tabrepo.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabrepo.benchmark.result import ExperimentResults
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils import load_results
from tabrepo.benchmark.models.model_register import tabrepo_model_register
from autogluon.core.models import AbstractModel
import yaml

from autogluon.common.utils.log_utils import set_logger_verbosity
set_logger_verbosity(2)  # Set logger so more detailed logging is shown for tutorial

def get_benchmark_metadata(benchmark: str, subset: list[str]=None) -> pd.DataFrame:
    if benchmark == "TabZilla":
        from tabprep.tabprep.utils import get_benchmark_dataIDs, get_metadata_df
        tids, dids = get_benchmark_dataIDs("TabZilla")
        metadata = get_metadata_df(tids, dids)    
    elif benchmark == "Grinsztajn":
        from tabprep.tabprep.utils import get_benchmark_dataIDs, get_metadata_df
        tids, dids = get_benchmark_dataIDs("Grinsztajn")
        metadata = get_metadata_df(tids, dids)    
    elif benchmark == "TabArena":
        metadata = load_task_metadata()        
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if subset is not None:
        metadata = metadata[metadata["name"].isin(subset)]

    return metadata

def get_model(model_name: str) -> AbstractModel:
    return tabrepo_model_register.key_to_cls(model_name)

def get_model_configs(model_name: str, n_trials: int) -> dict[str, Any]:
    use_configs = {model_name + ' (default)': {}}  # Number of configurations to use for the model
    if n_trials > 0:
        from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
        tabarena_context = TabArenaContext()
        mod_configs = tabarena_context.load_configs_hyperparameters([model_name], download='auto')  # load hyperparameters for the methods
        # TODO: Correctly sort HPs
        for num, (key,value) in enumerate(mod_configs.items()):
            if num >= n_trials:
                break
            use_configs[key] = {key_: value_ for key_, value_ in mod_configs[key]['hyperparameters'].items() if key_ != "ag_args_ensemble"}
    
    return use_configs

if __name__ == '__main__':
    with open("tabprep/base_config.yaml", 'r') as stream:
        configs = yaml.safe_load(stream)

    # my_exp_name = "openml"
    # input_format = "openml"
    # benchmark = "Grinsztajn"
    # use_configs = 10
    # ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    expname = str(".." / Path(__file__).parent / "experiments" / configs['benchmark']['name'] / configs['exp_name'])  # folder location to save all experiment artifacts
    repo_dir = str(".." / Path(__file__).parent / "repos" / configs['benchmark']['name'] / configs['exp_name'])  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`
    # expname = '/home/ubuntu/cat_detection/experiments/TabArena/current_test2'
    # repo_dir = '/home/ubuntu/cat_detection/repos/TabArena/current_test2'
    
    task_metadata = get_benchmark_metadata(configs['benchmark']['name'], subset=configs['benchmark'].get('subset', None))
    datasets = [i for i in list(task_metadata["name"])]


    if configs['benchmark']['outer_folds'] == "full":
        folds = list(range(1))
        repeats = list(range(1))  # Use all folds in the benchmark, and repeat the experiments for each repeat
    else:
        folds = list(range(configs['benchmark']['outer_folds']))
        repeats = list(range(configs['benchmark']['repeats']))


    my_model = get_model(configs['model']['name'])
    mod_configs = get_model_configs(configs['model']['name'], configs['hpo']['n_trials'])

    method_suffix = ""
    if configs['preprocessing']['use_ftd']:
        method_suffix = "_ftd"
    if configs['preprocessing']['input_format'] == "csv":
        method_suffix += "_csv"

    experiment_kwargs={
        "input_format": configs['preprocessing']['input_format'],
        "benchmark_name": configs['benchmark']['name']
    }
    # if configs['benchmark']['name'] == 'Grinsztajn':
    #     # Grinsztajn benchmark uses the `benchmark_name` to load the task metadata
    #     experiment_kwargs["train_size"] = 10000
    #     experiment_kwargs["test_size"] = 50000

    # This list of methods will be fit sequentially on each task (dataset x fold)
    # methods = [
    #     # This will be a `config` in EvaluationRepository, because it computes out-of-fold predictions and thus can be used for post-hoc ensemble.
    #     AGModelBagExperiment(  # Wrapper for fitting a single bagged model via AutoGluon
    #         # The name you want the config to have
    #         name = f"{method_name}A{method_suffix}",

    #         # The class of the model. Can also be a string if AutoGluon recognizes it, such as `"GBM"`
    #         # Supports any model that inherits from `autogluon.core.models.AbstractModel`
    #         model_cls=my_model,
            # model_hyperparameters={
    #             # "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},  # uncomment to fit folds sequentially, allowing for use of a debugger
    #             # 'bagging_fraction': 0.9292925595304,
    #             # 'bagging_freq': 1,
    #             # 'cat_l2': 0.2043927060013,
    #             # 'cat_smooth': 2.2692497770756,
    #             # 'extra_trees': False,
    #             # 'feature_fraction': 0.4037251099523,
    #             # 'lambda_l1': 0.6257740962955,
    #             # 'lambda_l2': 1.2674881653251,
    #             # 'learning_rate': 0.0864584339434,
    #             # 'max_cat_to_onehot': 90000,  #9,# 0.133393, 0.132144, 0.131133, 0.130612
                # 'min_data_in_leaf': 1,
    #             # 'min_data_per_group': 3,
    #             # 'num_leaves': 139,
    #             **config,  # The non-default model hyperparameters.
            # },  # The non-default model hyperparameters.

    #         num_bag_folds=8,  # num_bag_folds=8 was used in the TabArena 2025 paper
    #         time_limit=configs['benchmark']['time_limit'],  # time_limit=3600 was used in the TabArena 2025 paper
    #         experiment_kwargs=experiment_kwargs,
    #         method_kwargs={
    #             'preprocess_data': True,
    #             'preprocessor_name': 'default_csv',
    #             'fit_kwargs': {
    #                 'feature_generator': None
    #             }
    #         },
    #     )
    # for method_name, config in mod_configs.items()]
    
    from tabflow.cli.launch_jobs import JobManager
    methods_file = "configs_all_new.yaml"  # TODO: Need to create this file
    methods = JobManager.load_methods_from_yaml(methods_file=methods_file)
    methods = methods[:1]
    # from tabrepo.utils.config_utils import generate_bag_experiments
    # methods = generate_bag_experiments(model_cls=methods[0]['model_cls'], configs=methods, name_suffix_from_ag_args=True)
    methods = [AGModelBagExperiment(model_cls='GBM', 
                                    name=method['name'],
                                    # model_hyperparameters={
                                    #     "ag_args": {'name_suffix': '_r33', 'priority': -2},
                                    #     "bagging_fraction": 0.9625293420216,
                                    #     "bagging_freq": 1,
                                    #     "cat_l2": 0.1236875455555,
                                    #     "cat_smooth": 68.8584757332856,
                                    #     "extra_trees": False,
                                    #     "feature_fraction": 0.6189215809382,
                                    #     "lambda_l1": 0.1641757352921,
                                    #     "lambda_l2": 0.6937755557881,
                                    #     "learning_rate": 0.0154031028561,
                                    #     "max_cat_to_onehot": 17,
                                    #     "min_data_in_leaf": 1,
                                    #     "min_data_per_group": 30,
                                    #     "num_leaves": 68},
                                    model_hyperparameters={key: value for key, value in method['model_hyperparameters'].items() if key != "ag_args_ensemble"},
                                    # model_hyperparameters={'min_data_in_leaf': 2, **method['model_hyperparameters']},  # Ensure min_data_in_leaf is set to 1
                                    # model_hyperparameters={'min_data_in_leaf': 2, 'max_cat_to_onehot': 30000, **method['model_hyperparameters']},  # Ensure min_data_in_leaf is set to 1
                                    num_bag_folds=method['num_bag_folds'],
                                    time_limit=method['time_limit'],
                                    # method_kwargs=method['method_kwargs'],
                                    method_kwargs={
                                        'fit_kwargs': {
                                            'feature_generator': None
                                            }, 
                                            'preprocess_data': True, 
                                            'preprocessor_name': 'all_in_one'
                                            }
                                    ) for method in methods]  # Convert dicts to AGModelBagExperiment objects

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Get the run artifacts.
    # Fits each method on each task (datasets * folds)
    results_lst = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        repeats=repeats,
        methods=methods,
        ignore_cache=configs['ignore_cache'],
        raise_on_failure=True,
        # use_ftd=configs['preprocessing']['use_ftd'],  # Use FeatureTypeDetector to detect new categorical features
    )

    
    if configs['run_and_evaluate']:
        experiment_results = ExperimentResults(task_metadata=task_metadata)

        # Convert the run artifacts into an EvaluationRepository
        repo: EvaluationRepository = experiment_results.repo_from_results(results_lst=results_lst)
        repo.print_info()

        repo.to_dir(path=repo_dir)  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`

        print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

        # get the local results
        evaluator = Evaluator(repo=repo)
        metrics: pd.DataFrame = evaluator.compare_metrics().reset_index().rename(columns={"framework": "method"})

        # load the TabArena paper results
        tabarena_results: pd.DataFrame = load_results()
        tabarena_results = tabarena_results[[c for c in tabarena_results.columns if c in metrics.columns]]

        dataset_fold_map = metrics.groupby("dataset")["fold"].apply(set)

        def is_in(dataset: str, fold: int) -> bool:
            return (dataset in dataset_fold_map.index) and (fold in dataset_fold_map.loc[dataset])

        # filter tabarena_results to only the dataset, fold pairs that are present in `metrics`
        is_in_lst = [is_in(dataset, fold) for dataset, fold in zip(tabarena_results["dataset"], tabarena_results["fold"])]
        tabarena_results = tabarena_results[is_in_lst]

        metrics = pd.concat([
            metrics,
            tabarena_results,
        ], ignore_index=True)

        res_df = pd.DataFrame({dat: {method: np.mean(metrics.loc[np.logical_and(metrics.dataset==dat, metrics.method==method), "metric_error"].values) for method in metrics.method.unique()} for dat in metrics.dataset.unique()}).transpose()
        print(res_df)
        print(res_df[['GBM (default)', 'LightGBM_c1_BAG_L1', 'CAT (default)', 'AutoGluon 1.3 (4h)']])
        # if new_model_name in ["RealMLP", "RealMLP_priordetect", "RealMLP-original"]:
        #     print(res_df[[f"{new_model_name}", "REALMLP (default)",  "AutoGluon 1.3 (4h)",  "CAT (default)"]])
        # elif new_model_name in ["TabM", "TabM_priordetect", "TabM-original"]:
        #     print(res_df[[f"{new_model_name}", "TABM (default)",  "AutoGluon 1.3 (4h)",  "CAT (default)"]])
        # elif new_model_name in ["LightGBM", "LightGBM_priordetect", "LightGBM-original"]:
        #     print(res_df[[f"{new_model_name}", "GBM (default)",  "AutoGluon 1.3 (4h)",  "CAT (default)"]])
        # elif new_model_name in ["CatBoost", "CatBoost_priordetect", "CatBoost-original"]:
        #     print(res_df[[f"{new_model_name}", "CAT (default)",  "AutoGluon 1.3 (4h)",  "TABM (default)"]])

        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(f"Results:\n{metrics.head(100)}")

        calibration_framework = "RF (default)"

        from tabrepo.tabarena.tabarena import TabArena
        tabarena = TabArena(
            method_col="method",
            task_col="dataset",
            seed_column="fold",
            error_col="metric_error",
            columns_to_agg_extra=[
                "time_train_s",
                "time_infer_s",
            ],
            groupby_columns=[
                "metric",
                "problem_type",
            ]
        )

        leaderboard = tabarena.leaderboard(
            data=metrics,
            include_elo=True if configs['benchmark']['name'] == "TabArena" else False,
            elo_kwargs={
                "calibration_framework": calibration_framework,
            },
        )

        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(leaderboard)


