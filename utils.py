from __future__ import annotations
from numba import njit
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_1samp, wilcoxon, binomtest
import pickle

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import lightgbm as lgb
from statsmodels.stats.multitest import multipletests
from scipy.special import expit
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

    y_agg = y.groupby(X_use).mean()

    # If the no. of unique values is to high, the algorithm might take to long - we shrink to the shrink_u most frequent values
    if X_use.nunique()>shrink_u:

        y_agg_c = y.groupby(X_use).count()
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
    stats["target_group_mean_stds"] = float(y.groupby(X[col]).mean().std()) # How different the target is for this feature across different values
    stats["target_group_stds_mean"] = float(y.groupby(X[col]).std().mean()) # How different the target is for this feature inside a value
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
                means = y.groupby(X[col]).mean()
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
                means = y.groupby(X[col]).mean()
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

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class TargetMeanRegressor(BaseEstimator, RegressorMixin):
    """
    A regressor that learns the mean target value for each category
    in a single feature, and for unseen categories returns the
    overall target mean.
    """

    def __init__(self):
        # no hyperparameters—for smoothing you could add things like min_samples_leaf, smoothing, etc.
        pass

    def fit(self, X, y):
        """
        X : array-like or DataFrame, shape (n_samples, 1)
        y : array-like, shape (n_samples,)
        """
        # turn X into a 1-d pandas Series
        X_ser = self._to_series(X)
        y_arr = np.asarray(y)

        # compute per-category means
        df = pd.DataFrame({'feature': X_ser, 'target': y_arr})
        self.mapping_ = df.groupby('feature')['target'].mean().to_dict()
        # global mean for unseen categories
        self.global_mean_ = y_arr.mean()
        return self

    def predict(self, X):
        """
        X : array-like or DataFrame, shape (n_samples, 1)
        returns: array, shape (n_samples,)
        """
        X_ser = self._to_series(X)
        preds = [
            self.mapping_.get(val, self.global_mean_)
            for val in X_ser
        ]
        return np.asarray(preds)

    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("TargetMeanRegressor requires exactly one column")
            return X.iloc[:, 0]
        arr = np.asarray(X)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        if arr.ndim != 1:
            raise ValueError("X must be 1-d or a single-column 2-d array/DataFrame")
        return pd.Series(arr)
    


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class TargetMeanClassifier(ClassifierMixin, BaseEstimator):
    """
    Like TargetMeanClassifier but vectorized—and now handles NaNs
    by filtering them out of the per-category counts and
    falling back to the global distribution.
    """

    def __init__(self):
        self._estimator_type = "classifier"

    def fit(self, X, y):
        # 1) unify to 1-d array
        X_arr = self._to_array(X)
        y_arr = np.asarray(y)

        # 2) classes ↔ indices
        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        n_classes = len(self.classes_)

        # 3) category codes (NaN → code = -1)
        cat = pd.Categorical(X_arr)
        self.categories_ = cat.categories
        cat_codes = cat.codes
        n_cats = len(self.categories_)

        # 4) only keep non-missing rows for the per-category table
        valid = cat_codes >= 0
        joint = cat_codes[valid] * n_classes + y_idx[valid]

        # 5) build the contingency table
        counts = np.bincount(
            joint,
            minlength=n_cats * n_classes
        ).reshape(n_cats, n_classes)

        # 6) normalize each row → P(class | category)
        row_sums = counts.sum(axis=1, keepdims=True)
        # (if a category has zero total, this will produce NaNs, but
        #  in practice every fitted category should appear at least once)
        self.proba_ = counts / row_sums

        # 7) compute the true global distribution over ALL y’s
        global_counts = np.bincount(y_idx, minlength=n_classes)
        self.global_proba_ = global_counts / global_counts.sum()

        return self

    def predict_proba(self, X):
        X_arr = self._to_array(X)
        # unseen (or NaN) → code = -1
        codes = pd.Categorical(X_arr, categories=self.categories_).codes

        # stack: [per-cat rows; global fallback row]
        extended = np.vstack([self.proba_, self.global_proba_])

        # fancy‐index: codes==-1 → last row → global
        return extended[codes]

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _to_array(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("FastTargetMeanClassifier requires exactly one column")
            return X.iloc[:, 0].values
        arr = np.asarray(X)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        if arr.ndim != 1:
            raise ValueError("X must be 1-d or a single-column 2-d array/DataFrame")
        return arr


# class TargetMeanClassifier(ClassifierMixin, BaseEstimator):
#     """
#     A classifier that learns, for each category in a single feature,
#     the empirical class-probability vector.  Unseen categories
#     fall back to the overall class distribution.
#     """

#     def __init__(self):
#         _estimator_type = "classifier"

#     def fit(self, X, y):
#         """
#         X : array-like or DataFrame, shape (n_samples, 1)
#         y : array-like of labels shape (n_samples,)
#         """
#         X_ser = self._to_series(X)
#         y_arr = np.asarray(y)
#         # store distinct classes
#         self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
#         n_classes = len(self.classes_)

#         # build mapping: category → proba vector
#         df = pd.DataFrame({'feature': X_ser, 'label': y_arr})
#         self.mapping_ = {}
#         # helper to map a label to its integer index
#         label_to_idx = {c: i for i, c in enumerate(self.classes_)}
#         for cat, grp in df.groupby('feature'):
#             counts = np.bincount(
#                 [label_to_idx[v] for v in grp['label']],
#                 minlength=n_classes
#             )
#             self.mapping_[cat] = counts / counts.sum()

#         # global distribution for unseen
#         global_counts = np.bincount(y_idx, minlength=n_classes)
#         self.global_proba_ = global_counts / global_counts.sum()
#         return self

#     def predict_proba(self, X):
#         """
#         returns array shape (n_samples, n_classes)
#         """
#         X_ser = self._to_series(X)
#         proba = [
#             self.mapping_.get(val, self.global_proba_)
#             for val in X_ser
#         ]
#         return np.vstack(proba)

#     def predict(self, X):
#         """
#         returns predicted class labels
#         """
#         proba = self.predict_proba(X)
#         idx = np.argmax(proba, axis=1)
#         return self.classes_[idx]

#     def _to_series(self, X):
#         # same helper as above
#         if isinstance(X, pd.DataFrame):
#             if X.shape[1] != 1:
#                 raise ValueError("TargetMeanClassifier requires exactly one column")
#             return X.iloc[:, 0]
#         arr = np.asarray(X)
#         if arr.ndim == 2 and arr.shape[1] == 1:
#             arr = arr.ravel()
#         if arr.ndim != 1:
#             raise ValueError("X must be 1-d or a single-column 2-d array/DataFrame")
#         return pd.Series(arr)
    

class TargetMeanRegressorNN(TargetMeanRegressor):
    def fit(self, X, y):
        super().fit(X, y)
        # store numeric array of seen categories
        # (will error if keys aren’t numeric)
        self._categories = np.array(sorted(self.mapping_.keys()), dtype=float)
        return self

    def predict(self, X):
        X_ser = self._to_series(X)
        preds = []
        for v in X_ser:
            if v in self.mapping_:
                preds.append(self.mapping_[v])
            else:
                # nearest‐neighbor fallback
                idx = np.abs(self._categories - float(v)).argmin()
                nearest = self._categories[idx]
                preds.append(self.mapping_[nearest])
        return np.asarray(preds)
    

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator

class MultiFeatureTargetMeanClassifier(ClassifierMixin, BaseEstimator):
    """
    Maintains one (n_cat+1 × n_classes) table per feature, pads them,
    stacks to shape (n_features, max_categories+1, n_classes), then
    does one giant advanced-index lookup in predict_proba.
    """
    def __init__(self):
        self._estimator_type = "classifier"
        self._nan_sentinel = object()

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # temporary storage
        mapping_list = []
        categories_list = []
        fallback_list = []

        # build each feature’s table
        for col in X.columns:
            ser = X[col].where(X[col].notna(), self._nan_sentinel)
            codes, uniques = pd.factorize(ser, sort=False)
            n_cat = len(uniques)

            # count matrix
            M = np.zeros((n_cat, n_classes), dtype=int)
            np.add.at(M, (codes, y_idx), 1)

            # normalize rows → P(y|cat)
            M = M / M.sum(axis=1, keepdims=True)

            # global fallback row
            global_p = np.bincount(y_idx, minlength=n_classes)
            global_p = global_p / global_p.sum()
            M_ext = np.vstack([M, global_p])  # shape (n_cat+1, n_classes)

            mapping_list.append(M_ext)
            categories_list.append(uniques)
            fallback_list.append(n_cat)

        # figure out how many rows we need to pad to:
        max_rows = max(n_cat + 1 for n_cat in fallback_list)

        # stack into one big 3D array: (n_features, max_rows, n_classes)
        P = np.zeros((n_features, max_rows, n_classes), dtype=float)
        for j, M_ext in enumerate(mapping_list):
            rows = M_ext.shape[0]
            P[j, :rows, :] = M_ext
            # fill any extra rows with that feature’s fallback row:
            if rows < max_rows:
                P[j, rows:, :] = M_ext[-1]

        # stash everything you need for predict:
        self.mapping_array_    = P
        self.categories_       = categories_list
        self.fallback_array_   = np.array(fallback_list, dtype=int)
        self.feature_index_    = np.arange(n_features)
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        # 1) sentinelize NaNs
        X_fill = X.where(X.notna(), self._nan_sentinel)

        # 2) build a (n_samples × n_features) code matrix in one shot
        codes = np.stack([
            idx.get_indexer(X_fill[col])
            for idx, col in zip(self.categories_, X_fill.columns)
        ], axis=1)
        # unseen → fallback row
        codes = np.where(codes < 0, self.fallback_array_[None, :], codes)

        # 3) single advanced‐index lookup:
        #    mapping_array_[feature, code, :] → shape (n_samples, n_features, n_classes)
        proba = self.mapping_array_[self.feature_index_[None, :], codes]
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        # pick the best class per sample, per feature
        winners = np.argmax(proba, axis=2)
        return self.classes_[winners]





class TargetMeanClassifier(ClassifierMixin, BaseEstimator):
    """Vectorized target-mean classifier with NaN-as-own-category support."""
    def __init__(self):
        self._estimator_type = "classifier"
        # unique object sentinel for NaNs
        self._nan_sentinel = object()

    def fit(self, X, y):
        # 1) Seriesify and sentinelize NaNs
        X_ser = self._to_series(X)
        X_filled = X_ser.where(X_ser.notna(), self._nan_sentinel)

        # 2) factorize X into integer codes (0…n_categories-1)
        codes, uniques = pd.factorize(X_filled, sort=False)
        self.categories_ = list(uniques)
        n_categories = len(uniques)

        # 3) encode y
        y_arr = np.asarray(y)
        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)
        n_classes = len(self.classes_)

        # 4) build count matrix by indexing
        count_matrix = np.zeros((n_categories, n_classes), dtype=int)
        # for each sample i, increment count_matrix[codes[i], y_idx[i]]
        np.add.at(count_matrix, (codes, y_idx), 1)

        # 5) convert to per‐category probabilities
        #    – each row sums to 1
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        self.mapping_matrix_ = count_matrix / row_sums

        # 6) global fallback
        global_counts = np.bincount(y_idx, minlength=n_classes)
        self.global_proba_ = global_counts / global_counts.sum()

        # 7) stack fallback row for fast predict_proba
        self.mapping_array_ext_ = np.vstack([
            self.mapping_matrix_,
            self.global_proba_
        ])

        self._categories_index = pd.Index(self.categories_, dtype=object)
        self._fallback_idx    = n_categories

        return self


    def predict_proba(self, X):
        X_ser = self._to_series(X)
        X_filled = X_ser.where(X_ser.notna(), self._nan_sentinel)

        # vectorized lookup: returns –1 for unseen categories
        codes = self._categories_index.get_indexer(X_filled)
        # map all –1’s to the last row (global fallback)
        codes = np.where(codes < 0, self._fallback_idx, codes)
        # pull out the corresponding probability vectors
        return self.mapping_array_ext_[codes]

    def predict(self, X):
        proba = self.predict_proba(X)
        # choose class with highest probability
        winners = np.argmax(proba, axis=1)
        return self.classes_[winners]

    def _to_series(self, X):
        # same as your original helper
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("TargetMeanClassifier requires exactly one column")
            return X.iloc[:, 0]
        arr = np.asarray(X)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        if arr.ndim != 1:
            raise ValueError("X must be 1-d or a single-column 2-d array/DataFrame")
        return pd.Series(arr)



import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class PolynomialRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-style polynomial regression via closed-form OLS."""
    def __init__(self, degree=1):
        self.degree = degree

    def _design(self, X):
        x = np.asarray(X)
        if x.ndim == 2:
            if x.shape[1] != 1:
                raise ValueError("Expected a single feature.")
            x = x.ravel()
        elif x.ndim != 1:
            raise ValueError("X must be 1D or 2D with one feature.")
        # Build Vandermonde: columns [x^0, x^1, ..., x^degree]
        return np.vstack([x**d for d in range(self.degree + 1)]).T

    def fit(self, X, y):
        Xp = self._design(X)
        y = np.asarray(y)
        # Closed-form OLS: w = (X^T X)^{-1} X^T y
        XtX = Xp.T.dot(Xp)
        Xty = Xp.T.dot(y)
        w = np.linalg.solve(XtX, Xty)
        # Store intercept and coefficients
        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        Xp = self._design(X)
        return Xp.dot(np.concatenate([[self.intercept_], self.coef_]))


class PolynomialLogisticClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-style polynomial logistic regression via IRLS with ridge."""
    def __init__(self, degree=1, tol=1e-6, max_iter=100, reg=1e-6):
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        self.reg = reg  # L2 ridge strength

    def _design(self, X):
        x = np.asarray(X)
        if x.ndim == 2:
            if x.shape[1] != 1:
                raise ValueError("Expected a single feature.")
            x = x.ravel()
        elif x.ndim != 1:
            raise ValueError("X must be 1D or 2D with one feature.")
        # Build Vandermonde: columns [x^0, x^1, ..., x^degree]
        return np.vstack([x**d for d in range(self.degree + 1)]).T
    
    def fit(self, X, y):
        Xp = self._design(X)
        y = np.asarray(y)
        n, p1 = Xp.shape
        # Initialize weights vector
        w = np.zeros(p1)

        for _ in range(self.max_iter):
            z = Xp.dot(w)
            mu = expit(z) #1 / (1 + np.exp(-z))
            eps = 1e-8
            mu = np.clip(mu, eps, 1 - eps)
            W = mu * (1 - mu)
            # Prevent division by zero
            W_safe = np.where(W == 0, 1e-12, W)
            z_work = z + (y - mu) / W_safe

            # Weighted least squares components
            WX = W[:, None] * Xp
            H = Xp.T.dot(WX) + self.reg * np.eye(p1)
            rhs = Xp.T.dot(W * z_work)

            # Solve or fallback to pseudo-inverse
            try:
                w_new = np.linalg.solve(H, rhs)
            except np.linalg.LinAlgError:
                w_new = np.linalg.pinv(H).dot(rhs)

            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                break
            w = w_new

        # Store intercept and coefficients
        self.intercept_ = w[0]
        self.coef_ = w[1:]
        return self

    def predict_proba(self, X):
        Xp = self._design(X)
        z = Xp.dot(np.concatenate([[self.intercept_], self.coef_]))
        p = expit(z)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class UnivariateLinearRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-style univariate linear regression using closed-form OLS."""
    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X)
        # Accept 1D or (n_samples, 1)
        if X.ndim == 2:
            if X.shape[1] != 1:
                raise ValueError("Expected a single feature.")
            x = X.ravel()
        elif X.ndim == 1:
            x = X
        else:
            raise ValueError("X must be 1D or 2D with one feature.")
        y = np.asarray(y)

        # Compute statistics
        x_mean = x.mean()
        y_mean = y.mean()
        S_xy = ((x - x_mean) * (y - y_mean)).sum()
        S_xx = ((x - x_mean) ** 2).sum()

        # Store parameters
        self.coef_ = np.array([S_xy / S_xx])
        self.intercept_ = y_mean - self.coef_[0] * x_mean
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 2:
            x = X.ravel()
        else:
            x = X
        return self.intercept_ + self.coef_[0] * x

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

class UnivariateThresholdClassifier(ClassifierMixin, BaseEstimator):
    """
    Univariate classifier that finds the threshold on a single feature
    maximizing 0/1 accuracy via sort-and-sweep in O(N log N).
    """
    def __init__(self):
        # threshold_ will be set in fit
        self.threshold_ = None

    def _check_X(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 2:
            if X_arr.shape[1] != 1:
                raise ValueError("Expected a single feature.")
            return X_arr[:, 0]
        elif X_arr.ndim == 1:
            return X_arr
        else:
            raise ValueError("X must be 1D or 2D with one feature.")

    def fit(self, X, y):
        x = self._check_X(X)
        y = np.asarray(y).ravel()
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched lengths: X has {x.shape[0]} samples, y has {y.shape[0]}")

        # sort x and align y
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        n = x_sorted.shape[0]

        # total positives
        P = y_sorted.sum()
        # cumulative sum of positives
        cumsum_y = np.cumsum(y_sorted)

        # negatives to the left of threshold after i: (i+1) - cumsum_y[i]
        neg_left = np.arange(1, n+1) - cumsum_y
        # positives to the right: P - cumsum_y[i]
        pos_right = P - cumsum_y

        # compute accuracy at each threshold (after each sorted point)
        accuracy = (neg_left + pos_right) / n
        best_i = np.argmax(accuracy)

        # choose threshold halfway between the two points
        if best_i < n - 1:
            t = (x_sorted[best_i] + x_sorted[best_i + 1]) / 2.0
        else:
            # threshold above max
            t = x_sorted[best_i] + 1e-8

        self.threshold_ = t
        return self

    def predict(self, X):
        x = self._check_X(X)
        # predict 1 if x > threshold, else 0
        return (x > self.threshold_).astype(int)

    def predict_proba(self, X):
        x = self._check_X(X)
        preds = self.predict(x)
        # return deterministic 0/1 probabilities
        return np.vstack([1 - preds, preds]).T



# import numpy as np
# from scipy.special import expit
# from sklearn.base import BaseEstimator, ClassifierMixin

class MultiFeatureUnivariateLogisticClassifier(ClassifierMixin, BaseEstimator):
    """
    Vectorized univariate logistic regression: fits a separate logistic model for each feature in parallel.

    Parameters
    ----------
    tol : float
        Convergence tolerance for parameter updates.
    max_iter : int
        Maximum number of IRLS iterations.
    reg : float
        L2 regularization strength added to the Hessian diagonal to avoid singularities.
    """
    def __init__(self, tol=1e-6, max_iter=100, reg=1e-8):
        self.tol = tol
        self.max_iter = max_iter
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Ensure 2D design matrix (n_samples, n_features)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D array.")
        n_samples, n_features = X.shape

        # Precompute constant-feature mask and fallback intercept
        eps = 1e-8
        feat_var = X.var(axis=0)
        mask_const = feat_var < eps
        # Fallback intercept is logit(mean(y))
        pbar = np.clip(y.mean(), eps, 1 - eps)
        intercept_const = np.log(pbar / (1 - pbar))

        # Initialize parameters (shape: n_features,)
        w0 = np.zeros(n_features)
        w1 = np.zeros(n_features)

        # IRLS loop, vectorized across features
        for _ in range(self.max_iter):
            # Linear predictor: shape (n_samples, n_features)
            z = w0[None, :] + X * w1[None, :]
            mu = expit(z)
            mu = np.clip(mu, eps, 1 - eps)

            # Weights and working response
            W = mu * (1 - mu)                      # (n_samples, n_features)
            z_work = z + (y[:, None] - mu) / W     # (n_samples, n_features)

            # Sufficient statistics per feature
            S0 = W.sum(axis=0)                     # (n_features,)
            Sx = (W * X).sum(axis=0)
            Sxx = (W * X * X).sum(axis=0)
            b0 = (W * z_work).sum(axis=0)
            b1 = (W * z_work * X).sum(axis=0)

            # Add regularization to Hessian diag
            Sxx_reg = Sxx + self.reg
            D = S0 * Sxx_reg - Sx * Sx

            # Newton updates
            w0_new = (b0 * Sxx_reg - b1 * Sx) / D
            w1_new = (b1 * S0 - b0 * Sx) / D

            # For constant features, enforce fallback model (slope=0)
            w0_new[mask_const] = intercept_const
            w1_new[mask_const] = 0.0

            # Check convergence (max change across features)
            delta = np.max(np.abs(w0_new - w0) + np.abs(w1_new - w1))
            w0, w1 = w0_new, w1_new
            if delta < self.tol:
                break

        self.intercept_ = w0
        self.coef_ = w1
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D array.")

        # Linear predictors for each feature
        z = self.intercept_[None, :] + X * self.coef_[None, :]
        p = expit(z)
        return np.stack([1 - p, p], axis=2)

    def predict(self, X):
        proba_pos = self.predict_proba(X)[:, :, 1]
        return (proba_pos >= 0.5).astype(int)


class UnivariateLogisticClassifier(ClassifierMixin, BaseEstimator):
    """
    Sklearn-style univariate logistic regression using weighted IRLS on unique feature values.
    """
    def __init__(self, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter

    def _check_X(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 2:
            if X_arr.shape[1] != 1:
                raise ValueError("Expected a single feature.")
            return X_arr[:, 0]
        elif X_arr.ndim == 1:
            return X_arr
        else:
            raise ValueError("X must be 1D or 2D with one feature.")

    def fit(self, X, y):
        x = self._check_X(X)
        y = np.asarray(y).ravel()

        # Aggregate by unique x values
        uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sum_y = np.bincount(inv, weights=y)

        # Initialize parameters
        w0, w1 = 0.0, 0.0
        eps = 1e-8

        for _ in range(self.max_iter):
            # Linear predictor on unique x's
            z = w0 + w1 * uniq
            mu = expit(z)
            mu = np.clip(mu, eps, 1 - eps)

            # Full-data weights and working response
            W_tot = counts * (mu * (1 - mu))
            z_work = z + (sum_y - counts * mu) / W_tot

            # Aggregated sufficient statistics
            S0 = W_tot.sum()
            Sx = (W_tot * uniq).sum()
            Sxx = (W_tot * uniq * uniq).sum()
            b0 = (W_tot * z_work).sum()
            b1 = (W_tot * z_work * uniq).sum()

            # Newton update (2x2 system)
            D = S0 * Sxx - Sx * Sx
            w0_new = (b0 * Sxx - b1 * Sx) / D
            w1_new = (b1 * S0 - b0 * Sx) / D

            # Check convergence
            if abs(w0_new - w0) + abs(w1_new - w1) < self.tol:
                w0, w1 = w0_new, w1_new
                break
            w0, w1 = w0_new, w1_new

        self.intercept_ = w0
        self.coef_ = np.array([w1])
        return self

    def predict_proba(self, X):
        x = self._check_X(X)
        z = self.intercept_ + self.coef_[0] * x
        p = expit(z)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


# @njit
# def fit_univar_numba(x, y, tol, max_iter):
#     w0 = 0.0
#     w1 = 0.0
#     n = x.shape[0]
#     eps = 1e-8
#     for _ in range(max_iter):
#         # gradient and Hessian accumulators
#         g0 = 0.0
#         g1 = 0.0
#         H00 = 0.0
#         H01 = 0.0
#         H11 = 0.0

#         for i in range(n):
#             xi = x[i]
#             yi = y[i]
#             zi = w0 + w1 * xi
#             # manual expit and clamp to avoid zero weights
#             mui = 1.0 / (1.0 + np.exp(-zi))
#             if mui < eps:
#                 mui = eps
#             elif mui > 1.0 - eps:
#                 mui = 1.0 - eps
#             wi = mui * (1.0 - mui)

#             zi_work = zi + (yi - mui) / wi

#             # accumulate gradient
#             g0 += wi * zi_work
#             g1 += wi * zi_work * xi
#             # accumulate Hessian
#             H00 += wi
#             H01 += wi * xi
#             H11 += wi * xi * xi

#         # solve 2×2 Newton system
#         D      = H00 * H11 - H01 * H01
#         w0_new = (g0 * H11 - g1 * H01) / D
#         w1_new = (g1 * H00 - g0 * H01) / D

#         # check for convergence
#         if abs(w0_new - w0) + abs(w1_new - w1) < tol:
#             w0, w1 = w0_new, w1_new
#             break

#         w0, w1 = w0_new, w1_new

#     return w0, w1

# class UnivariateLogisticClassifier(ClassifierMixin, BaseEstimator):
#     """Sklearn-style univariate logistic regression using IRLS."""
#     def __init__(self, tol=1e-6, max_iter=100):
#         self.tol = tol
#         self.max_iter = max_iter

#     def _check_X(self, X):
#         X = np.asarray(X)
#         if X.ndim == 2:
#             if X.shape[1] != 1:
#                 raise ValueError("Expected a single feature.")
#             return X.ravel()
#         elif X.ndim == 1:
#             return X
#         else:
#             raise ValueError("X must be 1D or 2D with one feature.")

#     def fit(self, X, y):
#         x_arr = np.asarray(X).ravel().astype(np.float64)
#         y_arr = np.asarray(y).ravel().astype(np.int64)  # or float64

#         # call the compiled routine
#         w0, w1 = fit_univar_numba(x_arr, y_arr, self.tol, self.max_iter)

#         self.intercept_ = w0
#         self.coef_      = np.array([w1])
#         return self

#     def predict_proba(self, X):
#         x = self._check_X(X)
#         z = self.intercept_ + self.coef_[0] * x
#         p = expit(z)
#         # Return shape (n_samples, 2)
#         return np.vstack([1 - p, p]).T

#     def predict(self, X):
#         proba = self.predict_proba(X)[:, 1]
#         return (proba >= 0.5).astype(int)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor

class NearestValueSmoother(BaseEstimator, TransformerMixin):
    """
    Smooths a pandas Series by replacing each value with the
    average of its k nearest values (including itself) in the training set,
    and returns a pandas Series preserving index and name.
    """

    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors to use for smoothing (must be >= 1).
        weights : {'uniform', 'distance'} or callable, default='uniform'
            Weight function used in prediction.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y=None):
        """
        Fit the internal KNN regressor on X (and y=None since we're regressing X onto itself).
        """
        self._is_series_ = isinstance(X, pd.Series)
        X_arr = self._validate_input(X)
        # regress x -> x
        self.knn_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        ).fit(X_arr, X_arr.ravel())
        return self

    def transform(self, X):
        """
        Replace each entry by the average of its k nearest neighbors.
        If X is a pandas Series, return a Series with the same index & name.
        Otherwise return a 2D numpy array of shape (n_samples, 1).
        """
        # remember original if it's a Series
        is_series = isinstance(X, pd.Series)
        name = X.name if is_series else None
        index = X.index if is_series else None

        X_arr = self._validate_input(X)
        smoothed = self.knn_.predict(X_arr).ravel()

        if is_series:
            return pd.Series(smoothed, index=index, name=name)
        else:
            return smoothed.reshape(-1, 1)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to X, then transform X, returning same type as transform().
        """
        return self.fit(X, y).transform(X)

    def _validate_input(self, X):
        """
        Convert X to a 2D numpy array of shape (n_samples, 1).
        """
        if isinstance(X, pd.Series):
            arr = X.values.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("Input DataFrame must have exactly one column")
            arr = X.values
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                arr = X.reshape(-1, 1)
            elif X.ndim == 2 and X.shape[1] == 1:
                arr = X
            else:
                raise ValueError("Input array must be 1D or a 2D single-column array")
        else:
            # fallback for array-like
            arr = np.asarray(X).reshape(-1, 1)
        return arr


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class BinnedWeightedMeanTransformer(BaseEstimator, TransformerMixin):
    """
    Bin each numeric column and replace values with the weighted mean
    of original values within each bin.
    """

    def __init__(self, n_bins=10, strategy='uniform'):
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X, y=None, sample_weight=None):
        X = pd.DataFrame(X).copy()
        n = len(X)
        # default to equal weights
        w = (sample_weight if sample_weight is not None 
             else np.ones(n))
        self.bin_edges_ = {}
        self.bin_means_ = {}

        for col in X.columns:
            # determine edges
            if self.strategy == 'quantile':
                _, edges = pd.qcut(X[col], q=self.n_bins,
                                   retbins=True, duplicates='drop')
            else:
                edges = np.linspace(X[col].min(), X[col].max(),
                                    self.n_bins + 1)
            self.bin_edges_[col] = edges
            # assign each sample to a bin index
            idx = np.digitize(X[col], edges[1:-1], right=True)
            # compute weighted mean per bin
            means = {}
            for b in range(len(edges) - 1):
                mask = (idx == b)
                if mask.any():
                    vals = X.loc[mask, col].to_numpy()
                    weights = np.array(w)[mask]
                    means[b] = np.average(vals, weights=weights)
                else:
                    means[b] = np.nan
            self.bin_means_[col] = means

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X_out = pd.DataFrame(index=X.index, columns=X.columns)

        for col in X.columns:
            edges = self.bin_edges_[col]
            idx = np.digitize(X[col], edges[1:-1], right=True)
            # map bin index → precomputed mean
            X_out[col] = [ self.bin_means_[col].get(b, np.nan)
                           for b in idx ]

        return X_out






import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score


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

def make_cv_scores_with_early_stopping(target_type, scorer, cv, early_stopping_rounds=20, 
                                       vectorized=False, verbose=False,
                                       groups=None
                                       ):
    """CV creation function for vectorized versions of the TargetEncoderModels."""
    def cv_scores_with_early_stopping(X_df, y_s, pipeline):
        scores = []
        for train_idx, test_idx in cv.split(X_df, y_s, groups=groups):
            X_tr, y_tr = X_df.iloc[train_idx], y_s.iloc[train_idx]
            X_te, y_te = X_df.iloc[test_idx], y_s.iloc[test_idx]

            final_model = pipeline.named_steps["model"]

            # if it's an LGBM model, pass in eval_set + callbacks
            if isinstance(final_model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                pipeline.fit(
                    X_tr, y_tr,
                    **{
                        "model__eval_set": [(X_te, y_te)],
                        "model__callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
                        # "model__verbose": False
                    }
                )
            else:
                # dummy / other estimators: plain fit
                pipeline.fit(X_tr, y_tr)

            # predict
            if target_type == "binary" and hasattr(pipeline, "predict_proba"):
                if vectorized:
                    preds = pipeline.predict_proba(X_te)[:, :, 1]
                else:
                    preds = pipeline.predict_proba(X_te)[:, 1]
            else:
                preds = pipeline.predict(X_te)

            if vectorized:
                scores.append({col: scorer(y_te, preds[:,num]) for num, col in enumerate(X_df.columns)})
            else:
                scores.append(scorer(y_te, preds))

        return np.array(scores)
    
    return cv_scores_with_early_stopping




