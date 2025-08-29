from __future__ import annotations
from matplotlib import scale
from numba import njit
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler, MinMaxScaler, PowerTransformer
import os
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_1samp, wilcoxon, binomtest
import pickle

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from statsmodels.stats.multitest import multipletests
from scipy.special import expit
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, log_loss, accuracy_score, f1_score, precision_score, recall_score
from tabprep.proxy_models import CustomLinearModel

def clean_feature_names(X_input: pd.DataFrame):
    X = X_input.copy()
    X.columns = [str(col).replace('[', '').replace(']', '').replace(':', '')
                                            .replace('<', '').replace('>', '')
                                            .replace('=', '').replace(',', '')
                                            .replace(' ', '_') for col in X.columns]
    
    return X


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

def get_benchmark_dataIDs(benchmark_name):
    import openml
    if benchmark_name == "Grinsztajn": # Use the most original version of each unique dataset
        # collection_id = 334 # Tabular benchmark categorical classification
        # collection_id = 335 # Tabular benchmark categorical regression
        # collection_id = 336 # Tabular benchmark numerical regression
        # collection_id = 337 # Tabular benchmark numerical classification
        joint_tids = []
        joint_dids = []
        unique_names = set()
        for collection_id in [335, 336,  334, 337]:
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
        return tasks, data

def p_value_sign_test_median_greater_than_zero(data):
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

def p_value_wilcoxon_greater_than_zero(data):
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if np.all(data == 0):
        return 1.0  # No difference from zero
    
    # Add for robustness to outliers
    if np.median(data)==0:
        return 1.0
    
    stat, p = wilcoxon(data, alternative='greater')
    return p

def p_value_ttest_greater_than_zero(data):
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


def load_dataset(task):
    import openml
    data = openml.datasets.get_dataset(task)
    print(data.name)
    # if data.name!="Bank_Customer_Churn":
    #     continue
    X, _, _, _ = data.get_data()
    y = X[data.default_target_attribute]
    X = X.drop(columns=[data.default_target_attribute])

    # if data.name in ["Amazon_employee_access", "Bank_Customer_Churn", "Bioresponse"]:
    if all(y.apply(lambda x: x in ["No", "Yes"])):
        y = y.map({"No": 0, "Yes": 1}).astype(float)
    elif all(y.apply(lambda x: x in ["Bad", "Good"])):
        y = y.map({"Bad": 0, "Good": 1}).astype(float)
    elif data.name == "bank-marketing":
        y = y.map({"no": 0, "yes": 1}).astype(float)
    elif data.name == "APSFailure":
        y = y.map({"neg": 0, "pos": 1}).astype(float)
    elif data.name == "anneal":
        y = (y=="3").astype(float)
    elif data.name == "credit-g":
        y = y.map({"good": 0, "bad": 1}).astype(float)
    elif data.name == "customer_satisfaction_in_airline":
        y = y.map({"satisfied": 0, "dissatisfied": 1}).astype(float)
    elif data.name == "electricity":
        y = y.map({"DOWN": 0, "UP": 1}).astype(float)
    # else:
    #     y = y.astype(float)
    # if y.nunique()<=10:
    #     y = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=y.name)

    # if y.nunique()>2 and collection_id == 379:
    #     y = (y==y.value_counts().index[0]).astype(float)

    # Focus on non-binary features
    print(f"Binary: {sum(X.nunique()<=2)}")
    X = X.loc[:, X.nunique()>2]

    # Focus on columns where at least one cell can be transformed to numeric
    print(f"Non-numeric: {sum(X.apply(pd.to_numeric, **{'errors': 'coerce'}).isna().sum()==X.shape[0])}")
    X = X.loc[:, X.apply(pd.to_numeric, **{'errors': 'coerce'}).isna().sum()<X.shape[0]]

    # Focus on columns with at least 10 unique values
    print(f"Low-cardinality: {sum(X.nunique()<10)}")
    X = X.loc[:, X.nunique()>=5]

    # Try to make partially coerced series numeric by cleaning common characters
    part_coerced = X.columns[[0<float(pd.to_numeric(X[col].dropna(), errors= 'coerce').isna().sum())<X.shape[0] for col in X.columns]]
    print(f"Partially coerced: {len(part_coerced)}")
    if len(part_coerced)>0:
        X_copy = X.loc[:, part_coerced].apply(clean_series)
        X = X.drop(columns=part_coerced)
        X[part_coerced] = X_copy
    
    X = X.astype(float)

    return X, y, data.name  


def get_feature_type_metadata(save_name="feature_type_metadata.pkl", return_dict=False):

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
    

def is_misclassified_numeric(series: pd.Series, threshold: float = 0.9) -> bool:
    """
    Detect if a pandas Series has been incorrectly formatted as a non-numeric
    type (object, string, or category) although it should be numeric.

    Parameters:
    - series: pd.Series to analyze
    - threshold: float between 0 and 1 indicating the minimum proportion of values 
                 that must be convertible to numbers to consider the series numeric

    Returns:
    - bool: True if the series is likely misclassified, False otherwise
    """
    if not pd.api.types.is_object_dtype(series) and \
       not pd.api.types.is_string_dtype(series) and \
       not pd.api.types.is_categorical_dtype(series):
        return False  # Already a numeric type

    # Try converting to numeric
    coerced = pd.to_numeric(series, errors='coerce')
    valid_ratio = coerced.notna().mean()

    return valid_ratio >= threshold

def p_value_mean_test(data):
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

def cohen_d(data, popmean=0):
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    return (np.mean(data) - popmean) / np.std(data, ddof=1)

def merge_low_frequency_values(series: pd.Series, freq: int) -> pd.Series:
    """
    Merges low-frequency values in a Series with adjacent values until all remaining
    values have frequency > freq. Merging is done based on numeric proximity and frequency.

    Parameters:
    - series: pd.Series containing numeric values.
    - freq: int, the minimum frequency threshold.

    Returns:
    - pd.Series with merged values.
    """
    series_cp = series.copy()

    while True:
        cnts = series_cp.value_counts().sort_index()
        low_freq_vals = cnts[cnts <= freq]

        if low_freq_vals.empty:
            break  # All values now meet the frequency threshold

        # Handle the first low-frequency value in the list
        for val, cnt in low_freq_vals.items():
            idx = cnts.index.get_loc(val)

            # Choose the nearest neighbor (left or right)
            if idx == 0:
                neighbor = cnts.index[1]
            elif idx == len(cnts) - 1:
                neighbor = cnts.index[-2]
            else:
                left = cnts.index[idx - 1]
                right = cnts.index[idx + 1]
                neighbor = left if cnts[left] >= cnts[right] else right

            neighbor_cnt = cnts[neighbor]
            merged_val = (val * cnt + neighbor * neighbor_cnt) / (cnt + neighbor_cnt)

            # Replace both current and neighbor values with merged value
            series_cp.replace({val: merged_val, neighbor: merged_val}, inplace=True)

            # Restart loop to recalculate value counts
            break

    return series_cp


def local_and_distant_means(x: pd.Series, y: pd.Series, k=5, 
                            max_k=5, smooth=False, shrink_u=1000, method="distant",
                            average_infrequent=0
                            ) -> pd.DataFrame:
    """
    For each value in the Series `y`, calculate:
    1. The mean of the k values closest by index.
    2. The mean of the k values farthest by index.
    
    Returns a DataFrame with columns: 'local_mean', 'distant_mean'
    """


    if average_infrequent>0:
        X_use = merge_low_frequency_values(x, average_infrequent)
    else:
        X_use = x

    y_agg = y.groupby(X_use, observed=False).mean()

    # If the no. of unique values is to high, the algorithm might take to long - we shrink to the shrink_u most frequent values
    if X_use.nunique()>shrink_u:

        y_agg_c = y.groupby(X_use, observed=False).count()
        y_agg = y_agg.loc[y_agg_c.sort_values(ascending=False).iloc[:shrink_u].index].sort_index()

    n = len(y_agg)

    if k>y_agg.shape[0]-k:
        print(f"Can't use k={k} for a column with {n} unique values. Shrink k to {int((n-1)/2)}" )
        k = int((n-1)/2)

    result = pd.DataFrame(index=y_agg.index, columns=['local_mean', 'distant_mean'])
    indices = y_agg.index.to_numpy()

    # max_k = np.min([max_k, y_agg.shape[0]-k]) # Unsure whether it exactly makes sense this way


    for i, idx in enumerate(indices):
        np.random.seed(i)
        distances = np.abs(indices - idx)
        sorted_idx = np.argsort(distances)
        
        # Exclude the current index from the distance calculation
        sorted_idx = sorted_idx[sorted_idx != i]

        if method=="distant":        
            if smooth:
                closest_indices = indices[sorted_idx[:k]]
                farthest_indices = indices[sorted_idx[-k:]]
            else:
                closest_indices = indices[sorted_idx[k]]
                farthest_indices = indices[sorted_idx[-(k+1)]]
        if method=="random":        
            closest_indices = indices[sorted_idx[:k]][-max_k:]
            farthest_indices = np.random.choice(indices[sorted_idx[k:]], k)[-max_k:]  

        local_mean = np.abs(y_agg.loc[closest_indices]-y_agg.iloc[i]).mean()
        distant_mean = np.abs(y_agg.loc[farthest_indices]-y_agg.iloc[i]).mean()

            
        result.at[idx, 'local_mean'] = local_mean
        result.at[idx, 'distant_mean'] = distant_mean

    return result



# def clean_series(x, freq_threshold=3):
#     series = x.copy()
#     # Tokenize non-numeric elements across all rows
#     pattern = re.compile(r'[^\d.,]+')  # Match any non-digit, non-dot, non-comma token

#     tokens = [token for x in series.dropna().unique() for token in pattern.findall(str(x))]
#     u,c = np.unique(tokens,return_counts=True)
#     u = u[c>freq_threshold]
#     if len(u)>0:
#         print(u)
#         for i in u:
#             series = series.apply(lambda x: x.strip(i) if x is not np.nan else np.nan)

#         return pd.to_numeric(series, errors= 'coerce')
#     else:
#         print(f"No common patterns in {x.name}")
#         return x

# def clean_series(x):
#     return pd.to_numeric(
#         x.astype(str).apply(lambda s: re.sub(r'[^\d.,]', '', s) if pd.notnull(s) else np.nan),
#         errors='coerce'
#     )


def clean_series(x):
    def parse_value(val):
        if pd.isnull(val):
            return np.nan
        s = str(val).strip()
        
        # Handle '<X' or '>X' patterns
        if re.match(r'^<\s*\d+[.,]?\d*$', s):
            num = re.sub(r'[^\d.,]', '', s)
            try:
                return float(num.replace(',', '')) - 1
            except:
                return np.nan
        elif re.match(r'^>\s*\d+[.,]?\d*$', s):
            num = re.sub(r'[^\d.,]', '', s)
            try:
                return float(num.replace(',', '')) + 1
            except:
                return np.nan
        else:
            # Default cleaning: remove all non-digit/dot/comma characters
            cleaned = re.sub(r'[^\d.,]', '', s)
            try:
                return float(cleaned.replace(',', ''))
            except:
                return np.nan

    return x.apply(parse_value)



def get_cat_stats(X, y, col, average_infrequent=0):
    stats = {}
    stats["N"] = X.shape[0]
    stats["abs_target_corr"] = float(abs(X[col].corr(y)))
    stats["mean_abs_corr"] = float(X.corr()[col].drop(col).abs().mean())
    stats["max_abs_corr"] = float(X.corr()[col].drop(col).abs().max())
    stats["min_abs_corr"] = float(X.corr()[col].drop(col).abs().min())
    stats["target_group_mean_stds"] = float(y.groupby(X[col], observed=False).mean().std()) # How different the target is for this feature across different values
    stats["target_group_stds_mean"] = float(y.groupby(X[col], observed=False).std().mean()) # How different the target is for this feature inside a value
    stats["ordinal"] = float(all(X[col]==X[col].round(0)))
    stats["unique"] = float(X[col].nunique())
    stats["min_unique_freq"] = float(X[col].value_counts().min())
    stats["max_unique_freq"] = float(X[col].value_counts().max())
    stats["avg_unique_freq"] = float(X[col].value_counts().mean())
    stats["regular_distances_round4"] = float(pd.Series(np.diff(sorted(X[col].unique()))).round(4).value_counts().iloc[0]/X[col].nunique())

    if average_infrequent>0:
        X_use = merge_low_frequency_values(X[col], average_infrequent)
    else:
        X_use = X[col]


    loc_dist1 = local_and_distant_means(X_use, y, k=1, method="distant", smooth=False)
    loc_dist2 = local_and_distant_means(X_use, y, k=2, method="distant", smooth=False)
    stats["local_and_distant_diff_1"] = float(loc_dist1.mean().diff().iloc[1])
    stats["local_and_distant_diff_2"] = float(loc_dist2.mean().diff().iloc[1])
    stats["local_and_distant_diff_3"] = float(local_and_distant_means(X_use, y, k=3, smooth=True).mean().diff().iloc[1])
    stats["local_and_distant_diff_4"] = float(local_and_distant_means(X_use, y, k=4, smooth=True).mean().diff().iloc[1])
    stats["local_and_distant_diff_5"] = float(local_and_distant_means(X_use, y, k=5, smooth=True).mean().diff().iloc[1])
    
    # stats["pvalue_locdist1"] = p_value_mean_test(loc_dist1.diff(axis=1).iloc[:,1].astype(float))
    # stats["cohend_locdist1"] = cohen_d(loc_dist1.diff(axis=1).iloc[:,1].astype(float))
    # stats["pvalue_locdist2"] = p_value_mean_test(loc_dist2.diff(axis=1).iloc[:,1].astype(float))
    # stats["cohend_locdist2"] = cohen_d(loc_dist2.diff(axis=1).iloc[:,1].astype(float))

    stats["wilcoxon_locdist1"] = p_value_wilcoxon_greater_than_zero(loc_dist1.diff(axis=1).iloc[:,1].astype(float))
    # stats["wilcoxon_locdist2"] = p_value_wilcoxon_greater_than_zero(loc_dist2.diff(axis=1).iloc[:,1].astype(float))

    # stats["signtest_locdist1"] = p_value_sign_test_median_greater_than_zero(loc_dist1.diff(axis=1).iloc[:,1].astype(float))
    # stats["signtest_locdist2"] = p_value_sign_test_median_greater_than_zero(loc_dist2.diff(axis=1).iloc[:,1].astype(float))

    return stats


def generate_category_report_pdf(X, y, save_path, dataset_name, show_corr_feature=True,                                  
                                 smooth=True, method="random",average_infrequent=0, 
                                 feat_limit=1000,
                                 
                                 ):
    """Generates a PDF report of categorical feature statistics and visualizations."""

    def draw_text_subplot(ax, df_col_stats, font_size=8):
        """Render a single-column DataFrame as monospaced text on a subplot."""
        df_str = df_col_stats.to_string()
        ax.axis('off')
        ax.text(0, 1, df_str, fontsize=font_size, family='monospace', va='top')

    if not os.path.exists(f'{save_path}'):
        os.mkdir(f'{save_path}')

    with PdfPages(f'{save_path}/{dataset_name}.pdf') as pdf:
        for col in X.columns:
            u = X[col].nunique()
            if u > feat_limit:
                print(f"Skipping {col} with {u} unique values") 
                continue

            # Stats for this column only
            col_stats_dict = get_cat_stats(X, y, col)
            col_stats = pd.DataFrame(col_stats_dict, index=[0]).T
            col_stats.columns = ["col"]
            col_stats = col_stats.round(2)

            # Relationship to the target
            max_k = min(50, int(u/2))
            loc_dist_dict = {k: local_and_distant_means(X[col], y, k=k, smooth=smooth, method=method, average_infrequent=average_infrequent)
                             for k in range(1, max_k)}

            df_locdist_mean = pd.DataFrame({k: loc_dist_dict[k].mean() for k in range(1, max_k)})
            df_locdist_std = pd.DataFrame({k: loc_dist_dict[k].std() for k in range(1, max_k)})

            fig, axes = plt.subplots(1, 5 if show_corr_feature else 4, figsize=(15, 5), 
                                     gridspec_kw={'width_ratios': [2, 2, 1, 1] if show_corr_feature else [2, 2, 1, 1]})
            draw_text_subplot(axes[0], col_stats)

            ax = axes[1]
            ax.errorbar(df_locdist_mean.columns, df_locdist_mean.loc["local_mean"], 
                        yerr=df_locdist_std.loc["local_mean"], fmt='-o', capsize=5)
            ax.errorbar(df_locdist_mean.columns, df_locdist_mean.loc["distant_mean"], 
                        yerr=df_locdist_std.loc["distant_mean"], fmt='-o', capsize=5)
            ax.legend(["Local", "Distant"])
            ax.set_title(f'Local vs Distant Effect for: {col}')
            ax.set_xlabel('k (number of neighbors)')
            ax.set_ylabel('Mean Difference ± Std Dev')

            if show_corr_feature:
                corr_col = X.corr()[col].drop(col).abs().idxmax()

                loc_dist_dict = {k: local_and_distant_means(X[col], X[corr_col], k=k, smooth=smooth, method=method, average_infrequent=average_infrequent)
                                 for k in range(1, max_k)}

                df_locdist_mean = pd.DataFrame({k: loc_dist_dict[k].mean() for k in range(1, max_k)})
                df_locdist_std = pd.DataFrame({k: loc_dist_dict[k].std() for k in range(1, max_k)})

                ax = axes[2]
                ax.errorbar(df_locdist_mean.columns, df_locdist_mean.loc["local_mean"], 
                            yerr=df_locdist_std.loc["local_mean"], fmt='-o', capsize=5)
                ax.errorbar(df_locdist_mean.columns, df_locdist_mean.loc["distant_mean"], 
                            yerr=df_locdist_std.loc["distant_mean"], fmt='-o', capsize=5)
                ax.legend(["Local", "Distant"])
                ax.set_title(f'Local vs Distant Effect for: {corr_col}')
                ax.set_xlabel('k (number of neighbors)')
                ax.set_ylabel('Mean Difference ± Std Dev')

                # Mean-per-value plots
                means = y.groupby(X[col], observed=False).mean()
                axes[3].scatter(means.index, means.values)
                axes[3].set_title(f'Mean of y grouped by {col}')
                axes[3].set_xlabel(col)
                axes[3].set_ylabel('Mean y')

                # axes[4].hist(X[col], bins = 50)
                # axes[4].set_title(f'Distribution of {col}')
                # axes[4].set_xlabel(col)

                axes[4].boxplot(loc_dist_dict[1].diff(axis=1).iloc[:,1].astype(float))
                axes[4].set_title(f'Distribution of {col}')
                axes[4].set_xlabel(col)

            else:
                means = y.groupby(X[col], observed=False).mean()
                axes[2].scatter(means.index, means.values)
                axes[2].set_title(f'Mean of y grouped by {col}')
                axes[2].set_xlabel(col)
                axes[2].set_ylabel('Mean y')
                
                # axes[3].hist(X[col], bins = 50)
                # axes[3].set_title(f'Distribution of {col}')
                # axes[3].set_xlabel(col)


                axes[3].boxplot(loc_dist_dict[1].diff(axis=1).iloc[:,1].astype(float))
                axes[3].set_title(f'Distribution of {col}')
                axes[3].set_xlabel(col)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


import lightgbm as lgb

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, r2_score, root_mean_squared_error, roc_auc_score


def categorical_accuracy_diff(x, y, target_type):
    """
    Compare LightGBM performance using numeric vs categorical treatment for each feature.

    Parameters:
        X (pd.DataFrame): Feature dataframe
        y (array-like): Target variable

    Returns:
        dict: Dictionary where each key is a feature name and value is a dictionary with:
              - 'diff': mean accuracy difference (cat - normal)
              - 'normal_mean': mean accuracy (numeric)
              - 'normal_std': std accuracy (numeric)
              - 'cat_mean': mean accuracy (categorical)
              - 'cat_std': std accuracy (categorical)
    """
    scorer = make_scorer(roc_auc_score) if target_type=="binary" else make_scorer(lambda x,y: -root_mean_squared_error(x,y)) #make_scorer(r2_score)

    if target_type=="binary":
        model_class = lgb.LGBMClassifier
    else:
        model_class = lgb.LGBMRegressor
    
    params = {
        "objective": "binary" if target_type=="binary" else "regression",
        "learning_rate": 0.1,
        "max_depth": 1,
        "n_estimators": 100,
        "boosting_type": "gbdt",
        "verbosity": -1
    }

    scores = {}

    for mode in ["normal", "cat"]:
        x_use = x.copy()
        if mode == "cat":
            x_use = x_use.astype("category")
        # if mode == "normal":
        #     X_input = x.to_frame()
        # else:
        #     X_input = pd.get_dummies(x)

        X_input = x_use.to_frame()
        X_input.columns = [str(col).replace('[', '').replace(']', '').replace(':', '')
                                          .replace('<', '').replace('>', '')
                                          .replace('=', '').replace(',', '')
                                          .replace(' ', '_') for col in X_input.columns]
        model = model_class(**params)
        cv_scores = cross_val_score(model, X_input, y, cv=5, scoring=scorer, )

        scores[mode] = np.array(cv_scores)
        

    return scores

def deranged_series(s, random_state=None):
    n = len(s)
    rng = np.random.default_rng(random_state)

    # Start with a shuffled index
    shuffled_idx = s.sample(frac=1, random_state=random_state).index.to_numpy()
    fixed_points = (shuffled_idx == np.arange(n))

    # Fix any fixed points
    if fixed_points.any():
        fixed_indices = np.where(fixed_points)[0]

        for i in range(len(fixed_indices)):
            idx1 = fixed_indices[i]
            # Try to swap with the next one in the list (circularly)
            idx2 = fixed_indices[(i + 1) % len(fixed_indices)]
            shuffled_idx[idx1], shuffled_idx[idx2] = shuffled_idx[idx2], shuffled_idx[idx1]

    return s.iloc[shuffled_idx].reset_index(drop=True)


def categorical_accuracy_derange(x, y, target_type):
    """
    Compare LightGBM performance using numeric vs categorical treatment for each feature.

    Parameters:
        X (pd.DataFrame): Feature dataframe
        y (array-like): Target variable

    Returns:
        dict: Dictionary where each key is a feature name and value is a dictionary with:
              - 'diff': mean accuracy difference (cat - normal)
              - 'normal_mean': mean accuracy (numeric)
              - 'normal_std': std accuracy (numeric)
              - 'cat_mean': mean accuracy (categorical)
              - 'cat_std': std accuracy (categorical)
    """
    scorer = make_scorer(roc_auc_score) if target_type=="binary" else make_scorer(lambda x,y: -root_mean_squared_error(x,y)) #make_scorer(r2_score)

    if target_type=="binary":  
        model_class = lgb.LGBMClassifier
    else:
        model_class = lgb.LGBMRegressor
    
    params = {
        "objective": "binary" if target_type=="binary" else "regression",
        "learning_rate": 0.1,
        "max_depth": 1,
        "n_estimators": 100,
        "boosting_type": "gbdt",
        "verbosity": -1
    }

    scores = {}

    for mode in ["normal", "deranged"]:
        x_use = x.copy()
        if mode == "deranged":
            unique_x = pd.Series(x_use.unique())
            val_map = dict(zip(unique_x, deranged_series(unique_x)))
            x_use = x.map(val_map)
        # if mode == "normal":
        #     X_input = x.to_frame()
        # else:
        #     X_input = pd.get_dummies(x)

        X_input = x_use.to_frame()
        X_input.columns = [str(col).replace('[', '').replace(']', '').replace(':', '')
                                          .replace('<', '').replace('>', '')
                                          .replace('=', '').replace(',', '')
                                          .replace(' ', '_') for col in X_input.columns]
        model = model_class(**params)
        cv_scores = cross_val_score(model, X_input, y, cv=5, scoring=scorer)

        scores[mode] = np.array(cv_scores)
        

    return scores



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

def detect_cat_lgb_derange(x, y, target_type, sig_method="hard", verbose=False):
    scores = categorical_accuracy_derange(x, y, target_type)

    mean = np.mean(scores['normal']-scores['deranged'])
    std = np.std(scores['normal']-scores['deranged'])

    if sig_method=="hard":
        criterion = (mean-std)<=0.
    elif sig_method == "wilcoxon":
        diff_values = scores['normal']-scores['deranged']
        signtest_locdist1 = p_value_wilcoxon_greater_than_zero(diff_values)

        criterion = signtest_locdist1>0.05 and mean<0.1 # The latter is necessary as sometimes, if there is a lot of noise, single folds get high random values
    
    if verbose:
        print(f"{x.name}: {mean:3f} ({std:3f})")

    if criterion:
        return "nominal"
    else:
        return "numeric"

def detect_cat_lgb(x, y, target_type, verbose=False, sig_method="hard"):
    scores = categorical_accuracy_diff(x, y, target_type)

    mean = np.mean(scores['normal']-scores['cat'])
    std = np.std(scores['normal']-scores['cat'])
    
    if verbose:
        print(f"{x.name}: {mean:3f} ({std:3f})")
    
    if sig_method=="hard":
        criterion = (mean+std)<0.
    elif sig_method == "wilcoxon":
        diff_values = scores['normal']-scores['cat']
        signtest_locdist1 = p_value_wilcoxon_greater_than_zero(diff_values)

        criterion = signtest_locdist1>0.05
        
    if criterion:
    # if np.sign(loc_dist1.diff(axis=1).iloc[:,1]).median()<0.01:
        return "nominal"
    else:
        return "numeric"
    
def detect_cat_ordinal(x, **kwargs):
    if all(x==x.round(0)):
        return "nominal"
    else:
        return "numeric"
    
def detect_cat_reg_dist(x, round=2, thresh = 0.9, **kwargs):
    if float(pd.Series(np.diff(sorted(x.unique()))).round(round).value_counts().iloc[0]/x.nunique())>thresh:
        return "nominal"
    else:
        return "numeric"
    
def detect_cat_loc_dist(x, y, k=1, method="random", smooth=True, sig_method="hard", verbose=False):
    loc_dist1 = local_and_distant_means(x, y, k=k, method=method, smooth=smooth, shrink_u=1000)
    float(loc_dist1.mean().diff().iloc[1])
    
    
    if sig_method == "hard":
        criterion = loc_dist1.diff(axis=1).iloc[:,1].mean() <= 0
        if verbose:
            print(x.name, loc_dist1.diff(axis=1).iloc[:,1].mean())
    elif sig_method == "wilcoxon":
        diff_values = loc_dist1.diff(axis=1).iloc[:,1].astype(float)
        diff_values = diff_values.loc[diff_values.between(diff_values.quantile(0.1),diff_values.quantile(0.9))]
        signtest_locdist1 = p_value_wilcoxon_greater_than_zero(diff_values)

        criterion = signtest_locdist1>0.05
        if verbose:
            print(x.name, round(signtest_locdist1,2))

    if criterion:
        return "nominal"
    else:
        return "numeric"
    
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

def analyze_feature_types(X, y, target_type='binary', early_stopping_rounds=20, test_method = "lgb-based"):
    # set up
    if target_type == "binary":
        scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
        model_class = lgb.LGBMClassifier
        dummy = DummyClassifier(strategy='prior')
        linear = UnivariateLogisticClassifier()
        poly_2 = PolynomialLogisticClassifier(degree=2)
        poly_3 = PolynomialLogisticClassifier(degree=3)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
        model_class = lgb.LGBMRegressor
        dummy = DummyRegressor(strategy='mean')
        linear = UnivariateLinearRegressor()
        poly_2 = PolynomialRegressor(degree=2)
        poly_3 = PolynomialRegressor(degree=3)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    modes = [
        "normal", "cat", "deranged", 
        "mean", #"mean-NN", 
        "binned-mean", "linear", "poly2", "poly3"
        ] # "quantile-mean"]
    base_params = {
        "objective": "binary" if target_type=="binary" else "regression",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "verbosity": -1
    }
    res_df = pd.DataFrame(index=["dummy"] + modes)
    significances = {}
    assignments = {}

    def cv_scores_with_early_stopping(X_df, y_s, pipeline):
        scores = []
        for train_idx, test_idx in cv.split(X_df, y_s):
            X_tr, y_tr = X_df.iloc[train_idx], y_s.iloc[train_idx]
            X_te, y_te = X_df.iloc[test_idx], y_s.iloc[test_idx]

            final_model = pipeline.named_steps["model"]

            # if it's an LGBM model, pass in eval_set + callbacks
            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                pipeline.fit(
                    X_tr, y_tr,
                    **{
                        "model__eval_set": [(X_te, y_te)],
                        "model__callbacks": [lgb.early_stopping(early_stopping_rounds)],
                        # "model__verbose": False
                    }
                )
            else:
                # dummy / other estimators: plain fit
                pipeline.fit(X_tr, y_tr)

            # predict
            if target_type == "binary" and hasattr(pipeline, "predict_proba"):
                preds = pipeline.predict_proba(X_te)[:, 1]
            else:
                preds = pipeline.predict(X_te)

            scores.append(scorer(y_te, preds))
        return np.array(scores)

    # full‐X
    params = base_params.copy()
    params["max_bin_by_feature"] = (X.nunique()+1).tolist()
    full_pipe = Pipeline([("model", model_class(**params))])
    full_scores = cv_scores_with_early_stopping(X, y, full_pipe)
    res_df["full"] = round(full_scores.mean(), 4)

    # per‐column
    for cnum, col in enumerate(X.columns):
        print(cnum, col)
        significances[col] = {}
        scores = {}

        # dummy baseline on single column
        dummy_pipe = Pipeline([("model", dummy)])
        scores["dummy"] = cv_scores_with_early_stopping(X[[col]], y, dummy_pipe)

        # # full model on single column
        # model_pipe = Pipeline([("model", model_class(**base_params))])
        # scores["full"] = cv_scores_with_early_stopping(X[[col]], y, model_pipe)

        for mode in modes:
            # build mode-specific model & data
            params = base_params.copy()
            params["max_bin"] = min([10000, X[col].nunique()]) #if mode=="cat" else 2
            params["max_depth"] = 1 #if mode=="cat" else 2
            # params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)

            if mode in ("mean", "quantile-mean", "binned-mean"):
                model = (TargetMeanClassifier() if target_type=="binary"
                         else TargetMeanRegressor())
            elif mode in ("mean-NN"):
                model = (TargetMeanClassifierNN() if target_type=="binary"
                         else TargetMeanRegressorNN())
            elif mode in ("linear"):
                model = linear
            elif mode in ("poly2"):
                model = poly_2
            elif mode in ("poly3"):
                model = poly_3
            else:
                model = model_class(**params)

            x_use = X[col].copy()
            if mode == "deranged":
                uniques = pd.Series(x_use.unique())
                mapping = dict(zip(uniques, deranged_series(uniques, random_state=42)))
                x_use = x_use.map(mapping)
            if mode == "cat":
                x_use = x_use.astype("category")

            # pipeline
            if mode == "quantile-mean":
                from sklearn.preprocessing import QuantileTransformer
                pipe = Pipeline([
                    ("quantile", QuantileTransformer(
                        n_quantiles=int(x_use.nunique()*0.9),
                        output_distribution="uniform",
                        random_state=42
                    )),
                    ("model", model)
                ])
            elif mode in ["linear", "poly2", "poly3"]:
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")), 
                    ("standardize", StandardScaler(
                    )),
                    ("model", model)
                ])
            elif mode in ["binned-mean"]:
                pipe = Pipeline([
                    ("quantile-binned", BinnedWeightedMeanTransformer(n_bins=int(x_use.nunique()*0.75), strategy='quantile')),
                    ("model", model)
                ])
                

            else:
                pipe = Pipeline([("model", model)])

            X_mode = x_use.to_frame()
            scores[mode] = cv_scores_with_early_stopping(X_mode, y, pipe)

        # record means
        res_df[col] = {
            mode: round(scores[mode].mean(), 4)
            for mode in ["dummy"] + modes
        }

        # significance tests
        if test_method=="lgb-based":
            significances[col]["test_irrelevant_cont"] = p_value_wilcoxon_greater_than_zero(
                scores["dummy"] - scores["normal"]
            )
            significances[col]["test_irrelevant_cat"] = p_value_wilcoxon_greater_than_zero(
                scores["dummy"] - scores["cat"]
            )
            significances[col]["test_cat_superior"] = p_value_wilcoxon_greater_than_zero(
                scores["mean"] - scores["normal"]
            )
            significances[col]["test_num_superior"] = p_value_wilcoxon_greater_than_zero(
                scores["normal"] - scores["cat"]
            )
            significances[col]["test_derange_change"] = p_value_wilcoxon_greater_than_zero(
                scores["normal"] - scores["deranged"]
            )
        elif test_method == "interpolation":
            significances[col]["test_irrelevant_cont"] = p_value_wilcoxon_greater_than_zero(
                scores["dummy"] - scores["poly2"]
            )
            significances[col]["test_irrelevant_cat"] = p_value_wilcoxon_greater_than_zero(
                scores["dummy"] - scores["mean"]
            )
            significances[col]["test_cat_superior"] = p_value_wilcoxon_greater_than_zero(
                scores["mean"] - np.max([scores["linear"],scores["poly2"],scores["poly3"]],axis=0)
            )
            significances[col]["test_num_superior"] = p_value_wilcoxon_greater_than_zero(
                np.max([scores["linear"],scores["poly2"],scores["poly3"]],axis=0) - scores["mean"]
            )
            significances[col]["test_derange_change"] = 0. # p_value_wilcoxon_greater_than_zero(
            #     scores["normal"] - scores["deranged"]
            # )

    for col in X.columns:
        if significances[col]["test_irrelevant_cont"]<0.05 and significances[col]["test_irrelevant_cat"]<0.05:
            assignments[col] = "Irrelevant - user-defined"
            print(f"{col}: {assignments[col]}")
        elif significances[col]["test_cat_superior"]<0.05:
            assignments[col] = "Categorical"
            print(f"{col}: {assignments[col]}")
        elif significances[col]["test_num_superior"]<0.05 and significances[col]["test_derange_change"]<0.05:
            assignments[col] = "Numeric"
            print(f"{col}: {assignments[col]}")
        else:
            assignments[col] = "Both fine - User-defined"
            print(f"{col}: {assignments[col]}")

    assignments = pd.Series(assignments)

    if (assignments=="Categorical").any():
        X_new = X.copy()
        X_new[assignments[assignments == "Categorical"].index] = X_new.loc[:, assignments[assignments == "Categorical"].index].astype("category")

        params = base_params.copy()
        params["max_bin_by_feature"] = (X.nunique()+1).tolist()
        full_pipe = Pipeline([("model", model_class(**params))])
        full_scores = cv_scores_with_early_stopping(X_new, y, full_pipe)
        res_df["full_with_cat"] = round(full_scores.mean(), 4)


    return {
        "res_df": res_df,
        "assignments": pd.Series(assignments),
        "significances": significances
    }


def assign_closest(a_values: pd.Series, b_values: pd.Series) -> pd.Series:
    """
    For each value in series `a`, find the closest value in series `b`.
    
    Parameters:
    a (pd.Series): Series of values to match.
    b (pd.Series): Series of candidate values.
    
    Returns:
    pd.Series: Series of closest matches from b for each value in a.
    """
    # Convert to numpy for speed
    if isinstance(a_values, pd.Series):
        a_values = a_values.to_numpy()
    if isinstance(b_values, pd.Series):
        b_values = b_values.to_numpy()

    # Sort b for faster searching
    b_sorted = np.sort(b_values)
    
    # For each value in a, find position in sorted b where it could be inserted
    idxs = np.searchsorted(b_sorted, a_values, side="left")
    
    # Compare with neighbor on the left and right, choose the closest
    closest = []
    for val, idx in zip(a_values, idxs):
        candidates = []
        if idx > 0:
            candidates.append(b_sorted[idx - 1])
        if idx < len(b_sorted):
            candidates.append(b_sorted[min(idx, len(b_sorted) - 1)])
        closest_val = min(candidates, key=lambda x: abs(x - val))
        closest.append(closest_val)
    # FIXME: Check index
    return pd.Series(closest)

def make_cv_function(target_type, n_folds=5, early_stopping_rounds=20, 
                    random_state=42, 
                    vectorized=False, verbose=False,
                    groups=None,
                    drop_modes=0,
                                       ):
    """CV creation function for vectorized versions of the TargetEncoderModels."""
    if target_type=='binary':
        scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif target_type=='regression':
        scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif target_type=='multiclass':
        scorer = lambda y_true, y_pred: -log_loss(y_true, y_pred) # FIXME: Adjust to make sure function runs when not all classes are present
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:   
        raise ValueError("target_type must be 'binary' or 'regression'")

    def _density_weights(y_base, n_bins=20, clip=(0.2, 5.0)):
        vals = np.asarray(y_base)
        # Continuous targets: quantile-bin inverse frequency
        if np.issubdtype(vals.dtype, np.number) and np.unique(vals).size > max(10, n_bins // 2):
            from sklearn.preprocessing import KBinsDiscretizer
            kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            bins = kb.fit_transform(vals.reshape(-1, 1)).astype(int).ravel()
            counts = np.bincount(bins, minlength=n_bins).astype(float)
            w = 1.0 / np.maximum(counts[bins], 1.0)
        else:
            # Discrete labels: class inverse frequency
            _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
            w = 1.0 / np.maximum(counts[inv], 1.0)
        w = w / np.mean(w)
        if clip is not None:
            w = np.clip(w, clip[0], clip[1])
        return w

    def cv_func(X_df, y_s, pipeline, custom_prep=None, return_iterations=False, return_preds=False, 
                return_importances=False, scale_y=None, original_y=None, reg_assign_closest_y=False,
                sample_weights=None):
        scores = []
        iterations = []
        all_preds = []
        feature_importances = []
        if original_y is not None and scale_y != 'linear_residuals':
            splits = cv.split(X_df, original_y, groups=groups)
            scorer_use = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr)
        else:
            splits = cv.split(X_df, y_s, groups=groups)
            scorer_use = scorer
        for train_idx, test_idx in splits:
            X_tr, y_tr = X_df.iloc[train_idx], y_s.iloc[train_idx]
            X_te, y_te = X_df.iloc[test_idx], y_s.iloc[test_idx]
            
            col_names = X_df.columns

            if custom_prep is not None:
                for prep in custom_prep:
                    X_tr = prep.fit_transform(X_tr, y_tr)
                    X_te = prep.transform(X_te)
                if not isinstance(X_tr, pd.DataFrame):
                    if X_tr.shape[1]==len(col_names):
                        X_tr = pd.DataFrame(X_tr, columns=X_df.columns)
                    else:
                        X_tr = pd.DataFrame(X_tr)
                if not isinstance(X_te, pd.DataFrame):
                    if X_te.shape[1]==len(col_names):
                        X_te = pd.DataFrame(X_te, columns=X_df.columns)
                    else:
                        X_te = pd.DataFrame(X_te)

            if scale_y is not None:
                if scale_y  == 'standard':
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    y_tr = pd.Series(scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel(), name=y_tr.name, index=y_tr.index)
                    y_te = pd.Series(scaler.transform(y_te.values.reshape(-1, 1)).ravel(), name=y_te.name, index=y_te.index)
                elif scale_y  == 'power':
                    scaler = PowerTransformer()
                    y_tr = pd.Series(scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel(), name=y_tr.name, index=y_tr.index)
                    y_te = pd.Series(scaler.transform(y_te.values.reshape(-1, 1)).ravel(), name=y_te.name, index=y_te.index)
                elif scale_y  == 'quantile':
                    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
                    y_tr = pd.Series(scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel(), name=y_tr.name, index=y_tr.index)
                    y_te = pd.Series(scaler.transform(y_te.values.reshape(-1, 1)).ravel(), name=y_te.name, index=y_te.index)
                elif scale_y  == 'log':
                    y_tr = np.log1p(y_tr)
                    y_te = np.log1p(y_te)
                elif scale_y  == 'exp':
                    y_tr = np.expm1(y_tr)
                    y_te = np.expm1(y_te)
                elif scale_y == 'linear_residuals':
                    lm = CustomLinearModel(target_type)
                    lm.fit(X_tr, y_tr)
                    y_tr_lin = lm.predict(X_tr)
                    y_te_lin = lm.predict(X_te)
                else:
                    raise ValueError("scale_y must be 'standard' or 'log'")
            
            final_model = pipeline.named_steps["model"]

            if sample_weights == 'density':
                if original_y is not None:
                    y_tr_for_weights = original_y.iloc[train_idx]
                else:
                    y_tr_for_weights = y_s.iloc[train_idx]
                w_tr = _density_weights(y_tr_for_weights)                
            elif sample_weights is None:
                w_tr = None
            else:
                raise ValueError("sample_weights must be None or 'density'")

            # if it's an LGBM model, pass in eval_set + callbacks
            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                if scale_y == 'linear_residuals':
                    # if scale_y is linear_residuals, we need to pass the residuals
                    pipeline.fit(
                        X_tr, y_tr - y_tr_lin,
                        **fit_params,
                        **{
                            "model__eval_set": [(X_te, y_te - y_te_lin)],
                            "model__callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                            "model__sample_weight": w_tr,
                            # "model__verbose": False
                        }
                    )
                else:
                    pipeline.fit(
                        X_tr, y_tr,
                        **{
                            "model__eval_set": [(X_te, y_te)],
                            "model__callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                            "model__sample_weight": w_tr,
                            # "model__verbose": False
                        }
                    )
            else:
                # dummy / other estimators: plain fit
                pipeline.fit(X_tr, y_tr)

            # predict
            if target_type == "regression" or scale_y == 'linear_residuals' or not hasattr(pipeline, "predict_proba"):
                preds = pipeline.predict(X_te)
            elif target_type == "binary":
                if vectorized:
                    preds = pipeline.predict_proba(X_te)[:, :, 1]
                else:
                    preds = pipeline.predict_proba(X_te)[:, 1]
            elif target_type == 'multiclass':
                if vectorized:
                    preds = pipeline.predict_proba(X_te)
                else:
                    preds = pipeline.predict_proba(X_te)

            if scale_y is not None:
                if scale_y in ['standard', 'power', 'quantile']:
                    preds = pd.Series(scaler.inverse_transform(preds.reshape(-1, 1)).ravel(), name=y_te.name, index=y_te.index)
                    y_tr = pd.Series(scaler.inverse_transform(y_tr.values.reshape(-1, 1)).ravel(), name=y_tr.name, index=y_tr.index)
                    y_te = pd.Series(scaler.inverse_transform(y_te.values.reshape(-1, 1)).ravel(), name=y_te.name, index=y_te.index)
                elif scale_y == 'log':
                    preds = np.expm1(preds)
                    y_tr = np.expm1(y_tr)
                    y_te = np.expm1(y_te)
                elif scale_y  == 'exp':
                    preds = np.log1p(preds)
                    y_tr = np.log1p(y_tr)
                    y_te = np.log1p(y_te)
                elif scale_y == 'linear_residuals':
                    if target_type == 'binary':
                        preds = preds + y_te_lin
                        preds = preds.clip(0.00001,0.9999)
                else:
                    raise ValueError("scale_y must be 'standard' or 'log'")

            if reg_assign_closest_y and target_type == 'regression':
                # Assign closest y value to y_tr and y_te
                preds = assign_closest(preds,y_tr)


            if vectorized:
                scores.append({col: scorer_use(y_te, preds[:,num]) for num, col in enumerate(X_df.columns)})
            else:
                scores.append(scorer_use(y_te, preds))

            if return_preds:
                if isinstance(final_model, lgb.LGBMClassifier) and y_tr.nunique() > 2:
                    all_preds.append(pd.DataFrame(preds, index=y_te.index))
                else:
                    all_preds.append(pd.Series(preds, name=y_te.name, index=y_te.index))

            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                iterations.append(pipeline.named_steps['model'].booster_.num_trees())
                feature_importances.append(
                    pd.Series({col: imp for col,imp in zip(X_tr.columns,pipeline.named_steps['model'].feature_importances_)})
                )
            elif hasattr(pipeline.named_steps['model'], "coef_"):
                if len (X_tr.columns)>1:
                    feature_importances.append(
                        pd.Series({col: imp for col,imp in zip(X_tr.columns,pipeline.named_steps['model'].coef_[0])})
                    )
                else:
                    feature_importances.append(
                        pd.Series({X_tr.columns[0]: pipeline.named_steps['model'].coef_[0]})
                    )

        # TODO: Might change to return a dict instead
        if return_iterations and return_preds and return_importances:
            return np.array(scores), iterations, all_preds, feature_importances
        elif return_iterations and not return_preds and return_importances:
            return np.array(scores), iterations, feature_importances
        elif return_iterations and return_preds and not return_importances:
            return np.array(scores), iterations, all_preds
        elif return_iterations and not return_preds and not return_importances:
            return np.array(scores), iterations
        
        if not return_iterations and return_preds and return_importances:
            return np.array(scores), all_preds, feature_importances
        elif not return_iterations and not return_preds and return_importances:
            return np.array(scores), feature_importances
        elif not return_iterations and return_preds and not return_importances:
            return np.array(scores), all_preds
        elif not return_iterations and not return_preds and not return_importances:
            return np.array(scores)
        else:
            return np.array(scores)

    return cv_func


def drop_infrequent_values(series, thresh=1):
    value_counts = series.value_counts()
    non_unique_values = value_counts[value_counts > thresh].index
    return series[series.isin(non_unique_values)]

def drop_frequent_values(series, thresh=1):
    value_counts = series.value_counts()
    non_unique_values = value_counts[value_counts <= thresh].index
    return series[series.isin(non_unique_values)]

def drop_mode_values(series, thresh=1):
    value_counts = series.value_counts()
    mode_values = value_counts.index[thresh:]
    return series[series.isin(mode_values)]
    
def safe_stratified_group_kfold(X, y, groups, n_splits=5, max_attempts=1):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    attempt = 0

    while attempt < max_attempts:
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            test_labels = y[test_idx]
            class_counts = Counter(test_labels)
            if len(class_counts) < 2:
                break  # This fold has only one class
        else:
            return sgkf#.split(X, y, groups)  # All folds are good
        
        attempt += 1

    print("Could not generate stratified group folds with both classes in all test sets.")
    return None  # If we reach here, it means we couldn't find a valid split

def grouped_interpolation_test(x,y, target_type, add_dummy=False):
    q = int(x.nunique())
    
    if target_type == 'binary':
        # cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        cv = safe_stratified_group_kfold(x, y, x, n_splits=5)
        if cv is None:
            return None
        cv_scores_with_early_stopping = make_cv_scores_with_early_stopping(
            "binary", early_stopping_rounds=20, 
            verbose=False, groups=x
            )

        lgb_model = LGBMClassifier(verbose=-1, n_estimators=q, random_state=42, max_bin=q, max_depth=2)
    elif target_type == 'regression':
        cv = GroupKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_with_early_stopping = make_cv_scores_with_early_stopping(
            "regression", early_stopping_rounds=20, 
            verbose=False, groups=x
            )

        lgb_model = LGBMRegressor(verbose=-1, n_estimators=q, random_state=42, max_bin=q, max_depth=2)


    if add_dummy:
        pipe = Pipeline([("model", DummyRegressor())])
        res = cv_scores_with_early_stopping(x.to_frame(), y, pipe)
        print(f"Dummy: {np.mean(res):.4f} (+/- {np.std(res):.4f})")    

    pipe = Pipeline([("model", lgb_model)])
    res = cv_scores_with_early_stopping(x.astype(float).to_frame(), y, pipe)
    print(f"Performance: {np.mean(res):.4f} (+/- {np.std(res):.4f})")    
    return res


def analyze_cat_feature(x,y):
    # Insight: histograms look different depending on whether we treat a feature as string or float - could somehow use this information
    fig,ax = plt.subplots(1,4, figsize=(10, 5))
    ax[0].scatter(x, y)

    y_by_x = y.groupby(x, observed=False).mean()
    ax[1].scatter(y_by_x.index, y_by_x.values)

    ax[2].hist(x, bins=100)
    pd.Series(x.astype(float).sort_values()).plot(kind='hist', bins=100, ax=ax[3])

def get_feature_stats(x, y, target_type='binary', verbose=True):
    q = int(x.nunique())
    lgb_base_params = {
        "verbose": -1, "n_estimators": q*10, "random_state": 42, "max_bin": q, 
        "max_depth": 2, "min_samples_leaf": 1, "min_child_samples": 1,
    }
    stats = {}
    stats['q'] = q
    if target_type == 'binary':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_with_early_stopping = make_cv_scores_with_early_stopping(
            "binary", roc_auc_score, cv, early_stopping_rounds=20, 
            verbose=False
            )

        lgb_class = lambda lgb_params: LGBMClassifier(**lgb_params)
        target_model = TargetMeanClassifier()
        target_cut_model = lambda t: TargetMeanClassifierCut(q_thresh=t)
        lgb_cut_model = lambda t, p: LightGBMClassifierCut(q_thresh=t, init_kwargs=p)
    elif target_type == 'regression':
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_with_early_stopping = make_cv_scores_with_early_stopping(
            "regression", root_mean_squared_error, cv, early_stopping_rounds=20, 
            verbose=False
            )

        lgb_class = lambda lgb_params: LGBMRegressor(**lgb_params)
        target_model = TargetMeanRegressor()
        target_cut_model = lambda t: TargetMeanRegressorCut(q_thresh=t)
        lgb_cut_model = lambda t, p: LightGBMRegressorCut(q_thresh=t, init_kwargs=p)
    else:
        raise ValueError("Unsupported target type. Use 'binary' or 'regression'.")


    infreq = x.value_counts().sort_values(ascending=True).unique()[:100]

    # Target-based stats
    pipe = Pipeline([("model", target_model)])
    stats['mean'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)
    for t in infreq:
        t = int(t)

        pipe = Pipeline([("model", target_cut_model(t))])
        stats[f'mean-u>{t}'] = cv_scores_with_early_stopping(x.astype('category'), y, pipe)

        pipe = Pipeline([("model", target_model)])
        x_use = drop_infrequent_values(x,t)
        y_use = y.loc[x_use.index]
        stats[f'mean-drop-u<={t}'] = cv_scores_with_early_stopping(x_use.astype('category'), y_use, pipe)

        try:
            pipe = Pipeline([("model", target_model)])
            x_use = drop_frequent_values(x,t)
            y_use = y.loc[x_use.index]
            stats[f'mean-drop-u>{t}'] = cv_scores_with_early_stopping(x_use.astype('category'), y_use, pipe)
        except:
            continue
    # LGB-based stats
    pipe = Pipeline([("model", lgb_class(lgb_base_params))])
    stats['lgb-num'] = cv_scores_with_early_stopping(x.astype(float).to_frame(), y, pipe)
    pipe = Pipeline([("model", lgb_class(lgb_base_params))])
    stats['lgb-cat'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)
    for t in infreq:
        t = int(t)
        pipe = Pipeline([("model", lgb_cut_model(t, lgb_base_params))])
        stats[f'lgb-num-u>{t}'] = cv_scores_with_early_stopping(x.astype(float).to_frame(), y, pipe)

        pipe = Pipeline([("model", lgb_cut_model(t, lgb_base_params))])
        stats[f'lgb-cat-u>{t}'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)

    # pipe = Pipeline([("model", LightGBMClassifierCut(q_thresh=1, init_kwargs=lgb_base_params))])
    # stats['lgb-num-u>1'] = cv_scores_with_early_stopping(x.astype(float).to_frame(), y, pipe)
    # pipe = Pipeline([("model", LightGBMClassifierCut(q_thresh=2, init_kwargs=lgb_base_params))])
    # stats['lgb-num-u>2'] = cv_scores_with_early_stopping(x.astype(float).to_frame(), y, pipe)

    # pipe = Pipeline([("model", LightGBMClassifierCut(q_thresh=1, init_kwargs=lgb_base_params))])
    # stats['lgb-cat-u>1'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)
    # pipe = Pipeline([("model", LightGBMClassifierCut(q_thresh=2, init_kwargs=lgb_base_params))])
    # stats['lgb-cat-u>2'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)

    # for n_est in [int(q/2),int(q/4), int(q/8)]:
    #     p = lgb_base_params.copy()
    #     p["n_estimators"] = n_est
    #     pipe = Pipeline([("model", lgb_class(p))])
    #     stats[f'lgb-{n_est}est-num'] = cv_scores_with_early_stopping(x.astype(float).to_frame(), y, pipe)
    #     pipe = Pipeline([("model", lgb_class(p))])
    #     stats[f'lgb-{n_est}est-cat'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)

    if q>16:
        for m_bin in [int(q/2),int(q/4), int(q/8), int(q/16)]:
            # Cat features not affected 
            p = lgb_base_params.copy()
            p["max_bin"] = m_bin
            pipe = Pipeline([("model", lgb_class(p))])
            stats[f'lgb-{m_bin}bins-num'] = cv_scores_with_early_stopping(x.astype(float).to_frame(), y, pipe)


    # p = lgb_base_params.copy()
    # p["max_depth"] = 1
    # pipe = Pipeline([("model", lgb_class(p))])
    # stats['lgb-d1-cat'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)
    # pipe = Pipeline([("model", LightGBMClassifierCut(q_thresh=1, init_kwargs=p))])
    # stats['lgb-d1-cat-u>1'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)
    # pipe = Pipeline([("model", LightGBMClassifierCut(q_thresh=2, init_kwargs=p))])
    # stats['lgb-d1-cat-u>2'] = cv_scores_with_early_stopping(x.astype('category').to_frame(), y, pipe)


    if verbose:
        print(f"Feature: {x.name}")
        for key, value in stats.items():
            if key in ['q']:
                print(f"{key}: {value}")
            else:
                print(f"{key}: {np.mean(value):.4f} (+/- {np.std(value):.4f})")

    return stats
    
def sample_from_set(my_set, num_samples):
    if num_samples > len(my_set):
        raise ValueError("Number of samples requested exceeds the size of the set")
    my_list = list(my_set)
    return random.sample(my_list, num_samples)