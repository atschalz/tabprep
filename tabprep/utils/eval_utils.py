from __future__ import annotations

import openml
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon, ttest_1samp

from typing import Tuple, List

def get_benchmark_dataIDs(benchmark_name: str) -> Tuple[List[int], List[int]]:
    '''
    Retrieves the OpenML benchmark data and task IDs for a given benchmark name.
    For the Grinsztajn benchmark, it ensures that only the most original version of each unique dataset is used. The original benchmark consists of four sub-benchmarks separating regression vs. classification and datasets with vs. without categorical features. We don't support that currently since the datasets under the benchmark requirements don't contain complex categorical features and binarization of regression targets with reusing the datasets unnecessarily increases the no. of total datasets without information gain.
    Parameters:
    - benchmark_name (str): Name of the benchmark. Options are "Grinsztajn", "TabArena", "TabZilla".
    Returns:
    - tasks (list): List of OpenML task IDs.
    - data (list): List of OpenML dataset IDs.
    Raises:
    - ValueError: If the benchmark_name is not recognized.
    '''
    if benchmark_name == "Grinsztajn": # Use the most original version of each unique dataset
        joint_tids = []
        joint_dids = []
        unique_names = set()
        for collection_id in [335, 336,  334, 337]:
        # collection_id = 334: Tabular benchmark categorical classification
        # collection_id = 335: Tabular benchmark categorical regression
        # collection_id = 336: Tabular benchmark numerical regression
        # collection_id = 337: Tabular benchmark numerical classification
            benchmark_suite = openml.study.get_suite(collection_id)
            tids = benchmark_suite.tasks
            dids = benchmark_suite.data
            for did, tid in zip(dids,tids):
                task = openml.tasks.get_task(tid) 
                data = openml.datasets.get_dataset(did) 
                if data.name not in unique_names:
                    joint_tids.append(tid)
                    joint_dids.append(did)
                    unique_names.add(data.name)

        return joint_tids, joint_dids

    elif benchmark_name == "TabArena":
        collection_id = 457  # TabArena
        benchmark_suite = openml.study.get_suite(collection_id)
        tasks = benchmark_suite.tasks
        data = benchmark_suite.data
        return tasks, data

    elif benchmark_name == "TabZilla":
        collection_id = 379
        benchmark_suite = openml.study.get_suite(collection_id)
        tasks = benchmark_suite.tasks
        data = benchmark_suite.data

    else:
        raise ValueError(f"Unknown benchmark_name: {benchmark_name}")

    return tasks, data

def get_feature_type_metadata(save_name: str = "feature_type_metadata.pkl", 
                              return_dict: bool = False
                              ) -> dict:
    '''
    Retrieves and saves metadata about feature types for datasets in several OpenML benchmark suites, including TabArena, TabZilla, and the Grinsztajn benchmark. The metadata includes information about the benchmark name, task ID, dataset ID, task type, and feature types (both from OpenML and inferred from data types from converting to CSV). The metadata is saved as a pickle file.
    Args:
        save_name (str): Name of the file to save the metadata with path.
        return_dict (bool): If True, returns the metadata dictionary.
    Returns:
        dict: Metadata dictionary if return_dict is True.
    '''

    metadata = {}

    # Define the collection ID you want to access (e.g., OpenML100 benchmark suite)
    collection_ids = [
        457,  # TabArena
        334, # Tabular benchmark categorical classification
        335, # Tabular benchmark categorical regression
        336, # Tabular benchmark numerical regression
        337, # Tabular benchmark numerical classification
        379 # TabZilla-hard
    ]

    for cid in collection_ids:
        benchmark_suite = openml.study.get_suite(cid)
        tasks = benchmark_suite.tasks


        for tid in tasks[:]:
            task = openml.tasks.get_task(tid)
            data = task.get_dataset()
            X = data.get_data()[0]
            
            print(str(cid)+"__"+data.name)
            metadata[str(cid)+"__"+data.name] = {
                "benchmark": benchmark_suite.name,
                "bid": cid,
                "tid": tid,
                "did": data.id,
                "task": task.task_type,
                # "openml_feature_types": {i.name: i.data_type for i in data.features.values() if i.name!=task.target_name and i.name in X.columns}
            }

            metadata[str(cid)+"__"+data.name]["openml_feature_types"] = {key: "numeric" if ("int" in str(value) or "float" in str(value)) else "nominal" for key, value in dict(X.dtypes).items() if key!=task.target_name}

            X.to_csv("__TMPdata__.csv", index=False)
            X_new = pd.read_csv("__TMPdata__.csv")

            metadata[str(cid)+"__"+data.name]["csv_feature_types"] = {key: "numeric" if ("int" in str(value) or "float" in str(value)) else "nominal" for key, value in dict(X_new.dtypes).items() if key!=task.target_name}

            for f in metadata[str(cid)+"__"+data.name]["csv_feature_types"]:
                if X[f].dropna().nunique()==2:
                    metadata[str(cid)+"__"+data.name]["openml_feature_types"][f] = "binary"
                    metadata[str(cid)+"__"+data.name]["csv_feature_types"][f] = "binary"
    
    os.remove("__TMPdata__.csv")
    
    with open(save_name, 'wb') as file:
        pickle.dump(metadata, file)

    if return_dict:
        return metadata

def p_value_sign_test_median_greater_than_zero(data: np.array) -> float:
    """
    Perform a one-sided sign test to test if the median is greater than zero.

    Parameters:
        data (array-like): Numeric observations.

    Returns:
        float: One-sided p-value.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # Count how many values are > 0, < 0, or == 0
    n_pos = np.sum(data > 0)
    n_neg = np.sum(data < 0)
    n = n_pos + n_neg  # exclude zeros from the test

    if n == 0:
        return 1.0  # Cannot perform test

    # One-sided binomial test: H0: p <= 0.5 vs H1: p > 0.5
    result = binomtest(n_pos, n=n, p=0.5, alternative='greater')
    return result.pvalue

def p_value_wilcoxon_greater_than_zero(data: np.array) -> float:
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if np.all(data == 0):
        return 1.0  # No difference from zero
    
    # Add for robustness to outliers
    if np.median(data)==0:
        return 1.0
    
    stat, p = wilcoxon(data, alternative='greater')
    return p

def p_value_ttest_greater_than_zero(data: np.array) -> float:
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    
    if data.size == 0:
        raise ValueError("No data left after removing NaNs.")
    
    # If all values are exactly zero, no evidence for > 0
    if np.all(data == 0):
        return 1.0
    
    # Same median==0 check for robustness to outliers
    if np.median(data) == 0:
        return 1.0
    
    # Perform two‐sided t-test against popmean=0
    stat, p_two_sided = ttest_1samp(data, popmean=0, nan_policy='omit')
    
    # Convert to one‐tailed p for H1: mean > 0
    if stat > 0:
        return p_two_sided / 2
    else:
        # if t‐stat is ≤ 0, the one‐tailed p is 1 − (p_two_sided/2)
        return 1.0 - p_two_sided / 2
    
def p_value_mean_test(data: np.array) -> float:
    """
    Perform a one-sided one-sample t-test to check if the mean of the data is significantly > 0.

    Parameters:
        data (array-like): Numeric array of observations.

    Returns:
        float: One-sided p-value (for mean > 0).
    """
    data = np.asarray(data)
    
    # Remove NaNs if present
    data = data[~np.isnan(data)]

    if len(data) == 0:
        raise ValueError("Input data contains no valid (non-NaN) values.")

    t_stat, p_two_sided = ttest_1samp(data, popmean=0)

    # Convert two-sided p-value to one-sided (mean > 0)
    if t_stat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1.0  # Not significant in the right direction

    return p_one_sided

def cohen_d(data: np.array, popmean: int = 0) -> float:
    """
    Compute Cohen's d effect size.

    Parameters:
        data (array-like): Numeric array of observations.
        popmean (float): Population mean to compare against.

    Returns:
        float: Cohen's d value.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return (np.mean(data) - popmean) / np.std(data, ddof=1)

def nominal_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a similarity matrix for nominal (categorical) data based on the proportion of matching values across rows.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with nominal (categorical) values.
    
    Returns:
    pd.DataFrame: A square DataFrame with similarity scores (0 to 1), where 1 means perfect match.
    """
    cols = df.columns
    n = len(cols)
    sim_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    for col1 in cols:
        for col2 in cols:
            matches = (df[col1] == df[col2])
            similarity = matches.mean(skipna=True)  # Proportion of matches
            sim_matrix.loc[col1, col2] = similarity

    return sim_matrix

def true_nominal_rate(df: pd.DataFrame, positive_value="nominal") -> pd.DataFrame:
    """
    Computes a true negative rate matrix for binary nominal (string) data.
    
    Parameters:
    df (pd.DataFrame): DataFrame with string binary categorical values.
    positive_value (str): The value treated as the 'positive' class.

    Returns:
    pd.DataFrame: A square DataFrame with TNR scores.
    """
    cols = df.columns
    tnr_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    for col1 in cols:
        for col2 in cols:
            gt = df[col1]
            pred = df[col2]
            tn = ((gt != positive_value) & (pred != positive_value)).sum()
            fp = ((gt != positive_value) & (pred == positive_value)).sum()
            denom = tn + fp
            tnr_matrix.loc[col1, col2] = tn / denom if denom > 0 else 0.0

    return tnr_matrix
true_numeric_rate = lambda df: true_nominal_rate(df, positive_value="numeric")

def false_nominal_rate(df: pd.DataFrame, positive_value="nominal") -> pd.DataFrame:
    """
    Computes a false positive rate matrix for binary nominal (string) data.
    Replaces undefined FPR (due to 0 denominator) with 0.0.

    Parameters:
    df (pd.DataFrame): DataFrame with string binary categorical values.
    positive_value (str): The string value considered as the 'positive' class.

    Returns:
    pd.DataFrame: A square DataFrame with FPR scores.
    """
    cols = df.columns
    fpr_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    for col1 in cols:
        for col2 in cols:
            gt = df[col1]
            pred = df[col2]
            fp = ((gt != positive_value) & (pred == positive_value)).sum()
            tn = ((gt != positive_value) & (pred != positive_value)).sum()
            denom = fp + tn
            fpr_matrix.loc[col1, col2] = fp / denom if denom > 0 else 0.0

    return fpr_matrix

false_numeric_rate = lambda df: false_nominal_rate(df, positive_value="numeric")


def evaluate_binary_classification_metrics(y_true, y_pred, positive_label="nominal"):
    """
    Evaluate all relevant binary classification metrics for string labels.

    Args:
        y_true (list or array): Ground truth labels (strings)
        y_pred (list or array): Predicted labels (strings)
        positive_label (str): The label to consider as the "positive" class

    Returns:
        dict: Dictionary of all binary classification metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
        matthews_corrcoef,
        cohen_kappa_score,
        log_loss
    )
    from sklearn.preprocessing import LabelBinarizer

    results = {}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Validate classes
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    if len(unique_labels) != 2:
        raise ValueError(f"Binary classification expected, but found labels: {unique_labels}")
    if positive_label not in unique_labels:
        raise ValueError(f"Provided positive_label '{positive_label}' not found in labels: {unique_labels}")

    # Derive negative label
    negative_label = [label for label in unique_labels if label != positive_label][0]
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[negative_label, positive_label]).ravel()
    # results["confusion_matrix"] = {
    #     "true_negative": int(tn),
    #     "false_positive": int(fp),
    #     "false_negative": int(fn),
    #     "true_positive": int(tp)
    # }

    # Core metrics
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(y_true, y_pred, pos_label=positive_label)
    results["recall"] = recall_score(y_true, y_pred, pos_label=positive_label)
    results["specificity"] = tn / (tn + fp) if (tn + fp) else None  # specificity
    results["NPV"] = tn / (tn + fn) if (tn + fn) else None
    results["f1_score"] = f1_score(y_true, y_pred, pos_label=positive_label)
    results["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
    results["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    # Derived rates
    # results["true_positive_rate"] = tp / (tp + fn) if (tp + fn) else None  # recall
    # results["true_negative_rate"] = tn / (tn + fp) if (tn + fp) else None  # specificity
    # results["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else None
    # results["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else None

    # Binarize for AUC/log loss
    lb = LabelBinarizer(pos_label=1, neg_label=0)
    lb.fit([negative_label, positive_label])
    y_true_bin = lb.transform(y_true).ravel()
    y_pred_bin = lb.transform(y_pred).ravel()

    try:
        results["roc_auc"] = roc_auc_score(y_true_bin, y_pred_bin)
    except ValueError:
        results["roc_auc"] = None

    # try:
    #     results["log_loss"] = log_loss(y_true_bin, y_pred_bin)
    # except ValueError:
    #     results["log_loss"] = None

    # results["classification_report"] = classification_report(
    #     y_true, y_pred, labels=[negative_label, positive_label], output_dict=True
    # )

    return results