from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import random
from typing import Literal, Any, Dict

from tabprep.utils.eval_utils import p_value_wilcoxon_greater_than_zero
from sklearn.preprocessing import OrdinalEncoder

def cat_as_num(X_in):
    X_out = X_in.copy()
    for col in X_out.select_dtypes(include=['object', 'category']).columns:
        num_convertible = pd.to_numeric(X_out[col].dropna(), errors='coerce').notna().all()
        if num_convertible:
            X_out[col] = pd.to_numeric(X_out[col], errors='coerce')
        else:
            X_out[col] = OrdinalEncoder().fit_transform(X_out[[col]]).flatten()
    return X_out

def is_misclassified_numeric(series: pd.Series, threshold: float = 0.9) -> bool:
    # TODO: Seems to be unused, consider removing.
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

def merge_low_frequency_values(series: pd.Series, freq: int) -> pd.Series:
    # TODO: (1) Think about whether this is still needed; (2) Verify whether this works as intended; (3) Make it a sklearn preprocessor
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
    # TODO: (1) Verify whether this works as intended; (2) Move to eda
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
        rng = np.random.default_rng(i)
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
            farthest_indices = rng.choice(indices[sorted_idx[k:]], k)[-max_k:]  

        local_mean = np.abs(y_agg.loc[closest_indices]-y_agg.iloc[i]).mean()
        distant_mean = np.abs(y_agg.loc[farthest_indices]-y_agg.iloc[i]).mean()

            
        result.at[idx, 'local_mean'] = local_mean
        result.at[idx, 'distant_mean'] = distant_mean

    return result

def get_cat_stats(X, y, col, average_infrequent=0):
    # TODO: (1) Verify whether this works as intended; (2) Move to eda
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
    # TODO: (1) Verify whether this works as intended; (2) Move to eda

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

def deranged_series(s: pd.Series, random_state: int = None) -> pd.Series:
    '''
    Deranges the input Series by shuffling its values while ensuring that no value remains in its original position.
    Parameters:
        s (pd.Series): The input Series to be deranged.
        random_state (int, optional): Seed for the random number generator for reproducibility.
    Returns:
        pd.Series: A deranged version of the input Series.
    Raises:
        ValueError: If the input Series has less than 2 elements.
    '''
    # TODO: (1) Verify whether this works as intended; (2) Make it a sklearn preprocessor
    if len(s) < 2:
        raise ValueError("Series must have at least 2 elements to be deranged.")

    n = len(s)
    rng = np.random.default_rng(random_state)

    # Start with a shuffled index
    shuffled_idx = s.sample(frac=1, random_state=rng).index.to_numpy()
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

def assign_closest(a_values: pd.Series, b_values: pd.Series) -> pd.Series:
    """
    For each value in series `a`, find the closest value in series `b`.
    
    Parameters:
    a (pd.Series): Series of values to match.
    b (pd.Series): Series of candidate values.
    
    Returns:
    pd.Series: Series of closest matches from b for each value in a.
    """
    # TODO: Think whether we can make a useful preprocessor out of that
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


def drop_infrequent_values(series: pd.Series, thresh: int = 1) -> pd.Series:
    value_counts = series.value_counts()
    non_unique_values = value_counts[value_counts > thresh].index
    return series[series.isin(non_unique_values)]

def drop_frequent_values(series: pd.Series, thresh: int = 1) -> pd.Series:
    value_counts = series.value_counts()
    non_unique_values = value_counts[value_counts <= thresh].index
    return series[series.isin(non_unique_values)]

def drop_mode_values(series: pd.Series, thresh: int = 1) -> pd.Series:
    value_counts = series.value_counts()
    mode_values = value_counts.index[thresh:]
    return series[series.isin(mode_values)]

def grouped_interpolation_test(
        x: pd.Series,
        y: pd.Series, 
        target_type: str, 
        add_dummy: bool = False, 
        random_state: int = 42
        ) -> dict[str, Any]:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold
    from tabprep.utils.modeling_utils import make_cv_function, safe_stratified_group_kfold
    q = int(x.nunique())
    
    if target_type == 'binary':
        # cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv = safe_stratified_group_kfold(x, y, x, n_splits=5)
        if cv is None:
            return None
        cv_scores_with_early_stopping = make_cv_function(
            "binary", 
            verbose=False, groups=x
            )

        lgb_model = LGBMClassifier(verbose=-1, n_estimators=q, random_state=random_state, max_bin=q, max_depth=2)
    elif target_type == 'regression':
        cv = GroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores_with_early_stopping = make_cv_function(
            "regression", 
            verbose=False, groups=x
            )

        lgb_model = LGBMRegressor(verbose=-1, n_estimators=q, random_state=random_state, max_bin=q, max_depth=2)


    if add_dummy:
        pipe = Pipeline([("model", DummyRegressor())])
        res = cv_scores_with_early_stopping(x.to_frame(), y, pipe)
        print(f"Dummy: {np.mean(res):.4f} (+/- {np.std(res):.4f})")    

    pipe = Pipeline([("model", lgb_model)])
    res = cv_scores_with_early_stopping(x.astype(float).to_frame(), y, pipe)
    print(f"Performance: {np.mean(res):.4f} (+/- {np.std(res):.4f})")    
    return res

def sample_from_set(
        my_set: set, 
        num_samples: int
        ) -> list:
    if num_samples > len(my_set):
        raise ValueError("Number of samples requested exceeds the size of the set")
    my_list = list(my_set)
    return random.sample(my_list, num_samples)

def drop_highly_correlated_features(
        corr_matrix: pd.DataFrame, 
        threshold: float = 0.95
        ) -> pd.DataFrame:
    """
    Drops columns from the correlation matrix that are highly correlated (above threshold) with any other column,
    starting from the last column and moving backwards.

    Parameters:
    - corr_matrix (pd.DataFrame): The correlation matrix.
    - threshold (float): Correlation threshold above which columns will be dropped.

    Returns:
    - pd.DataFrame: Reduced correlation matrix with highly correlated columns dropped.
    """
    columns = corr_matrix.columns.tolist()

    i = len(columns) - 1
    while i >= 0:
        col = columns[i]
        # Check if this column is highly correlated with any of the others
        high_corr = corr_matrix[col].drop(labels=[col]).abs() > threshold
        if high_corr.any():
            # Drop the column
            corr_matrix = corr_matrix.drop(columns=col).drop(index=col)
            columns.pop(i)
        i -= 1

    return corr_matrix

def add_infrequent_category(s: pd.Series, min_count: int = 5, new_label: str = "Other") -> pd.Series:
    counts = s.value_counts()
    infreq = counts[counts < min_count].index

    if isinstance(s.dtype, pd.CategoricalDtype):
        if new_label not in s.cat.categories:
            s = s.cat.add_categories([new_label])
        s = s.where(~s.isin(infreq), new_label)
        s = s.cat.remove_unused_categories()
    else:
        s = s.where(~s.isin(infreq), new_label)

    return s