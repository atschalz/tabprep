from __future__ import annotations

import numpy as np
import openml
import pandas as pd
from tqdm import tqdm

def get_metadata_df(tasks, dids) -> pd.DataFrame:
    additional_metadata_df = []
    for task_id, did in tqdm(zip(tasks, dids)):
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        repeats, folds, _ = task.get_split_dimensions()

        target_feature = dataset.default_target_attribute
        is_classification = task.task_type == 'Supervised Classification'
        n_samples = dataset.qualities["NumberOfInstances"]
        n_features = dataset.qualities["NumberOfFeatures"]
        percentage_cat_features = dataset.qualities["PercentageOfSymbolicFeatures"]
        n_classes = dataset.qualities["NumberOfClasses"]

        if n_classes == 0:
            problem_type = "regression"
        else:
            problem_type = "binary" if n_classes == 2 else "multiclass"

        # From Paper
        if n_samples < 2_500:
            tabarena_repeats = 10
        elif n_samples > 250_000:  # Fallback
            tabarena_repeats = 1
        else:
            tabarena_repeats = 3

        # Bools for constrained methods from paper
        # - 10k training samples (so 2/3 * n_samples)
        can_run_tabpfnv2 = (
            (n_samples <= 15_000) and (n_features <= 500) and (n_classes <= 10)
        )
        # - n_classes != 0 as TabICL only works for classification
        can_run_tabicl = (
            (n_samples <= 150_000) and (n_features <= 500) and (n_classes != 0)
        )

        additional_metadata_df.append(
            [
                did,
                task_id,
                target_feature,
                is_classification,
                dataset.name,
                problem_type,
                n_features,
                n_samples,
                n_classes if n_classes != 0 else np.nan,
                percentage_cat_features,
                folds,
                repeats,
                tabarena_repeats,
                can_run_tabpfnv2,
                can_run_tabicl,
                # REF_MAPPING[dataset.name],
                # SOURCE_MAP[dataset.name],
                # DOMAIN_MAP[dataset.name],
                dataset.collection_date,
                dataset.licence,
                # URL_OVERWRITE_MAP.get(dataset.name, dataset.original_data_url),
            ],
        )

    additional_metadata_df = pd.DataFrame(
        additional_metadata_df,
        columns=[
            "dataset_id",
            "task_id",
            "target_feature",
            "is_classification",
            "openml_dataset_name",
            "problem_type",
            "num_features",
            "num_instances",
            "num_classes",
            "percentage_cat_features",
            "num_folds",
            "openml_num_repeats",
            "tabarena_num_repeats",
            "can_run_tabpfnv2",
            "can_run_tabicl",
            "reference",
            "data_source",
            "domain",
            "year",
            "licence",
            "original_data_url",
        ],
    )

    task_metadata["n_folds"] = 3
    task_metadata["n_repeats"] = task_metadata["NumberOfInstances"].apply(_get_n_repeats)
    task_metadata["n_features"] = (task_metadata["NumberOfFeatures"] - 1).astype(int)
    task_metadata["n_samples_test_per_fold"] = (task_metadata["NumberOfInstances"] / task_metadata["n_folds"]).astype(int)
    task_metadata["n_samples_train_per_fold"] = (task_metadata["NumberOfInstances"] - task_metadata["n_samples_test_per_fold"]).astype(int)

    task_metadata["dataset"] = task_metadata["name"]

    if subset is None:
        pass
    elif subset == "TabPFNv2":
        task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 10000]
        task_metadata = task_metadata[task_metadata["n_features"] <= 500]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] <= 10]
    elif subset == "TabICL":
        task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 100000]
        task_metadata = task_metadata[task_metadata["n_features"] <= 500]
        task_metadata = task_metadata[task_metadata["NumberOfClasses"] > 0]
    else:
        raise ValueError(f"Unknown subset: {subset}")
    return additional_metadata_df

