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


if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "current")  # folder location to save all experiment artifacts
    repo_dir = str(Path(__file__).parent / "repos" / "current")  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    task_metadata = load_task_metadata()

    # Sample for a quick demo
    # datasets = ["maternal_health_risk", "anneal", "customer_satisfaction_in_airline", "MIC", "airfoil_self_noise", "credit-g", "diabetes"]  # 
    # datasets = [i for i in list(task_metadata["name"]) if i not in ['superconductivity']]  # Exclude some datasets for a quick demo
    datasets = ['Bioresponse',
        'Food_Delivery_Time',
        'SDSS17',
        'airfoil_self_noise',
        'coil2000_insurance_policies',
        'customer_satisfaction_in_airline',
        'heloc',
        'students_dropout_and_academic_success',
        'superconductivity']
     
    # datasets = [#'Food_Delivery_Time', 
    #     'MIC', 
    #     'SDSS17', 
    #     'anneal', 
    # 'airfoil_self_noise',
    #    'kddcup09_appetency', 'coil2000_insurance_policies', 'Diabetes130US',
    #    'heloc', 'E-CommereShippingData', 'in_vehicle_coupon_recommendation',
    #    'Amazon_employee_access', 'churn', 'Fitness_Club',
    #    'online_shoppers_intention', 'credit_card_clients_default',
    #    'HR_Analytics_Job_Change_of_Data_Scientists', 'GiveMeSomeCredit',
    #    'customer_satisfaction_in_airline', 'Is-this-a-good-customer',
    #    'qsar-biodeg', 'Bioresponse', 
    # 'maternal_health_risk',
    #    'students_dropout_and_academic_success', 
    #    'superconductivity',
    # "website_phishing"
    # ]

    folds = [0,1,2]

    # TODO: Need to change the metadata after my stuff
    new_model_name =  "CatBoost_priordetect"

    if new_model_name in ["RealMLP-original", "RealMLP_priordetect"]:
        from tabrepo.benchmark.models.ag import RealMLPModel as my_model
    elif new_model_name in ["TabM-original", "TabM_priordetect"]:
        from tabrepo.benchmark.models.ag import TabMModel as my_model
    elif new_model_name in ["LightGBM-original", "LightGBM_priordetect"]:
        from autogluon.tabular.models import LGBModel as my_model
    elif new_model_name in ["CatBoost-original", "CatBoost_priordetect"]:
        from autogluon.tabular.models import CatBoostModel as my_model
    elif new_model_name=="RealMLP":
      from model_with_cat_detection import RealMLPModelWithCatDetection as my_model
    elif new_model_name=="TabM":
        from model_with_cat_detection import TabMModelWithCatDetection as my_model
    elif new_model_name=="LightGBM":
       from model_with_cat_detection import LGBModelWithCatDetection as my_model
    elif new_model_name=="CatBoost":
       from model_with_cat_detection import CatBoostModelWithCatDetection as my_model


    # import your model classes
    # from autogluon.tabular.models import CatBoostModel as my_model
    # from tabrepo.benchmark.models.ag import RealMLPModel as my_model
    # from model_with_cat_detection import RealMLPModelWithCatDetection as my_model
    # from model_with_cat_detection import LGBModelWithCatDetection as my_model


    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        # This will be a `config` in EvaluationRepository, because it computes out-of-fold predictions and thus can be used for post-hoc ensemble.
        AGModelBagExperiment(  # Wrapper for fitting a single bagged model via AutoGluon
            # The name you want the config to have
            name=f"{new_model_name}_catdetect",

            # The class of the model. Can also be a string if AutoGluon recognizes it, such as `"GBM"`
            # Supports any model that inherits from `autogluon.core.models.AbstractModel`
            model_cls=my_model,
            model_hyperparameters={
                # "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},  # uncomment to fit folds sequentially, allowing for use of a debugger
            },  # The non-default model hyperparameters.
            num_bag_folds=8,  # num_bag_folds=8 was used in the TabArena 2025 paper
            time_limit=360000,  # time_limit=3600 was used in the TabArena 2025 paper
        ),
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Get the run artifacts.
    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
        use_ftd=True,
    )

    # for res in results_lst:
    #     info = res["method_metadata"]["info"]["children_info"]
    #     print(f'{res["task_metadata"]["name"]}:') 
    #     print(f'Reassigned as categoricals: {[len(info[c]["new_categorical"]) for c in info]}')
    #     print(f'Reassigned as numerics: {[len(info[c]["new_numeric"]) for c in info]}')
    #     print("--"*20)

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

    if new_model_name=="RealMLP":
        print(res_df[[f"{new_model_name}_catdetect", "REALMLP (default)",  "AutoGluon 1.3 (4h)",  "CAT (default)"]])
    elif new_model_name=="TabM":
        print(res_df[[f"{new_model_name}_catdetect", "TABM (default)",  "AutoGluon 1.3 (4h)",  "CAT (default)"]])
    elif new_model_name in ["LightGBM", "LightGBM_priordetect"]:
        print(res_df[[f"{new_model_name}_catdetect", "GBM (default)",  "AutoGluon 1.3 (4h)",  "CAT (default)"]])
    elif new_model_name=="CatBoost":
        print(res_df[[f"{new_model_name}_catdetect", "CAT (default)",  "AutoGluon 1.3 (4h)",  "TABM (default)"]])

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
        include_elo=True,
        elo_kwargs={
            "calibration_framework": calibration_framework,
        },
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)


