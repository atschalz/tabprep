from __future__ import annotations

import numpy as np
# import openml
import pandas as pd
from tqdm import tqdm

def _get_n_repeats(n_instances: int) -> int:
    """
    Get the number of n_repeats for the full benchmark run based on the 2025 paper.
    If < 2500 samples, n_repeats = 10, else n_repeats = 3

    Parameters
    ----------
    n_instances: int

    Returns
    -------
    n_repeats: int
    """
    if n_instances < 2500:
        n_repeats = 10
    else:
        n_repeats = 3
    return n_repeats

def get_metadata_df(tasks, dids, subset: str = None) -> pd.DataFrame:
    import openml
    metadata_df = []
    for task_id, did in tqdm(zip(tasks, dids)):
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        repeats, folds, _ = task.get_split_dimensions()

        target_feature = dataset.default_target_attribute
        is_classification = task.task_type == 'Supervised Classification'
        ttid = task.task_type
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

        metadata_df.append(
            [
                did,
                task_id,
                ttid,
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

    metadata_df = pd.DataFrame(
        metadata_df,
        columns=[
            "did",
            "tid",
            "ttid",
            "target_feature",
            "is_classification",
            "name",
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
            # "reference",
            # "data_source",
            # "domain",
            "year",
            "licence",
            # "original_data_url",
        ],
    )

    metadata_df["n_folds"] = 10
    metadata_df["n_repeats"] = metadata_df["num_instances"].apply(_get_n_repeats)
    metadata_df["n_features"] = (metadata_df["num_features"] - 1).astype(int)
    metadata_df["n_samples_test_per_fold"] = (metadata_df["num_instances"] / metadata_df["n_folds"]).astype(int)
    metadata_df["n_samples_train_per_fold"] = (metadata_df["num_instances"] - metadata_df["n_samples_test_per_fold"]).astype(int)

    metadata_df["dataset"] = metadata_df["name"]

    if subset is None:
        pass
    elif subset == "TabPFNv2":
        metadata_df = metadata_df[metadata_df["n_samples_train_per_fold"] <= 10000]
        metadata_df = metadata_df[metadata_df["n_features"] <= 500]
        metadata_df = metadata_df[metadata_df["NumberOfClasses"] <= 10]
    elif subset == "TabICL":
        metadata_df = metadata_df[metadata_df["n_samples_train_per_fold"] <= 100000]
        metadata_df = metadata_df[metadata_df["n_features"] <= 500]
        metadata_df = metadata_df[metadata_df["NumberOfClasses"] > 0]
    else:
        raise ValueError(f"Unknown subset: {subset}")

    # Merge on Task ID
    # metadata_df = pd.merge(
    #     metadata_df, additional_metadata_df, on="task_id", how="left"
    # )
    #metadata_df.to_csv("metadata/tabarena_dataset_metadata.csv", index=False)
    return metadata_df

