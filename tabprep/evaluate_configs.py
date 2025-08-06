from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from tabrepo import EvaluationRepository, Evaluator, benchmark
from tabrepo.benchmark.experiment import AGModelBagExperiment, ExperimentBatchRunner
from tabrepo.benchmark.result import ExperimentResults
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils import load_results
import yaml


def get_benchmark_metadata(benchmark: str, subset: list[str]=None) -> pd.DataFrame:
    if benchmark == "TabZilla":
        from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
        tids, dids = get_benchmark_dataIDs("TabZilla")
        metadata = get_metadata_df(tids, dids)    
    elif benchmark == "Grinsztajn":
        from tabprep.utils import get_benchmark_dataIDs, get_metadata_df
        tids, dids = get_benchmark_dataIDs("Grinsztajn")
        metadata = get_metadata_df(tids, dids)    
    elif benchmark == "TabArena":
        metadata = load_task_metadata()        
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if subset is not None:
        metadata = metadata[metadata["name"].isin(subset)]

    return metadata

def get_model(model_name: str) -> Any:
    if model_name == "RealMLP":
        from tabrepo.benchmark.models.ag import RealMLPModel
        return RealMLPModel
    elif model_name == "TabM":
        from tabrepo.benchmark.models.ag import TabMModel
        return TabMModel
    elif model_name == "TabPFNV2":
        from tabrepo.benchmark.models.ag import TabPFNV2Model
        return TabPFNV2Model
    elif model_name == "LightGBM":
        from autogluon.tabular.models import LGBModel
        return LGBModel
    elif model_name == "CatBoost":
        from autogluon.tabular.models import CatBoostModel
        return CatBoostModel
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_model_configs(model_name: str, n_trials: int) -> dict[str, Any]:
    use_configs = {model_name + ' (default)': {}}  # Number of configurations to use for the model
    if n_trials > 0:
        from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
        tabarena_context = TabArenaContext()
        mod_configs = tabarena_context.load_configs_hyperparameters([model_name], download=True)  # load hyperparameters for the methods
        # TODO: Correctly sort HPs
        for num, (key,value) in enumerate(mod_configs.items()):
            if num >= n_trials:
                break
            use_configs[key] = {key_: value_ for key_, value_ in mod_configs[key]['hyperparameters'].items() if key_ != "ag_args_ensemble"}
    
    return use_configs

if __name__ == '__main__':
    with open("tabprep/base_config.yaml", 'r') as stream:
        configs = yaml.safe_load(stream)

    expname = str(Path(__file__).parent / "experiments" / configs['benchmark']['name'] / configs['exp_name'])  # folder location to save all experiment artifacts
    repo_dir = str(Path(__file__).parent / "repos" / configs['benchmark']['name'] / configs['exp_name'])  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`
    
    
    task_metadata = get_benchmark_metadata(configs['benchmark']['name'], subset=configs['benchmark'].get('subset', None))
    datasets = [i for i in list(task_metadata["name"])]

    if configs['benchmark']['outer_folds'] == "full":
        folds = [0,1,2]
    else:
        folds = list(range(configs['benchmark']['outer_folds']))

    results_lst = []
    for model_config in [
        ('LightGBM', False),
        ('LightGBM', True),                 
        ('TabM', False),                 
        ('TabM', True),                 
                         ]:

        model_name, use_ftd = model_config

        my_model = get_model(model_name)
        mod_configs = get_model_configs(model_name, configs['hpo']['n_trials'])

        method_suffix = ""
        if use_ftd:
            method_suffix = "_ftd"
        if configs['preprocessing']['input_format'] == "csv":
            method_suffix += "_csv"

        # This list of methods will be fit sequentially on each task (dataset x fold)
        methods = [
            # This will be a `config` in EvaluationRepository, because it computes out-of-fold predictions and thus can be used for post-hoc ensemble.
            AGModelBagExperiment(  # Wrapper for fitting a single bagged model via AutoGluon
                # The name you want the config to have
                name = f"{method_name}{method_suffix}",

                # The class of the model. Can also be a string if AutoGluon recognizes it, such as `"GBM"`
                # Supports any model that inherits from `autogluon.core.models.AbstractModel`
                model_cls=my_model,
                model_hyperparameters={
                    # "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},  # uncomment to fit folds sequentially, allowing for use of a debugger
                    **config,  # The non-default model hyperparameters.
                },  # The non-default model hyperparameters.

                num_bag_folds=configs['benchmark']['inner_folds'],  # num_bag_folds=8 was used in the TabArena 2025 paper
                time_limit=configs['benchmark']['time_limit'],  # time_limit=3600 was used in the TabArena 2025 paper
                experiment_kwargs={"input_format": configs['preprocessing']['input_format']},
            )
        for method_name, config in mod_configs.items()]
        
        exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

        # Get the run artifacts.
        # Fits each method on each task (datasets * folds)
        results_lst += exp_batch_runner.run(
            datasets=datasets,
            folds=folds,
            methods=methods,
            ignore_cache=configs['ignore_cache'],
            use_ftd=use_ftd,  # Use FeatureTypeDetector to detect new categorical features
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

        import matplotlib.pyplot as plt

        if 'LightGBM (default)_ftd' in res_df.columns:
            improvements = (res_df['LightGBM (default)']/res_df['LightGBM (default)_ftd'])-1
            improvements.sort_values().plot(kind='barh', figsize=(5, 10), colormap='coolwarm')
            plt.tight_layout()
            plt.savefig(f"{configs['benchmark']['name']}_improvements_lgb.png")
            plt.show()

        if 'TabM (default)_ftd' in res_df.columns:
            improvements = (res_df['TabM (default)']/res_df['TabM (default)_ftd'])-1
            improvements.sort_values().plot(kind='barh', figsize=(5, 10), colormap='coolwarm')
            plt.tight_layout()
            plt.savefig(f"{configs['benchmark']['name']}_improvements_tabm.png")
            plt.show()

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


