import numpy as np
import pandas as pd
from numba import njit, prange

from tabprep.preprocessors.type_change import CatAsNumTransformer

from typing import Literal, Optional
# import numexpr as ne
'''
Further filtering ideas:
- based on target_corr - if its the same, the feature is likely to contain the same info
'''


def remove_mostlynan_features(X):
    return X.loc[:, X.isna().mean() < 0.99]
def remove_constant_features(X):
    return X.loc[:, X.astype("float64").std() > 0] # float64 to avoid overflow warning

def basic_filter(
        X_in: pd.DataFrame, 
        # y_in: pd.Series,
        min_cardinality: int = 3,
        candidate_cols: list = None,
        use_polars: bool = False
        ) -> list:
    X = X_in.copy()
    
    # Filter by minimum cardinality
    X = X.loc[:, X.nunique() >= min_cardinality]

    if X.empty:
        return X

    # if predetermined candidate columns are given, use them
    if candidate_cols is not None:
        X = X[candidate_cols]

    if use_polars:
        from tabprep.preprocessors.arithmetic.filtering_polars import remove_mostlynan_features_pl, remove_constant_features_pl
        X = remove_mostlynan_features_pl(X)
        X = remove_constant_features_pl(X)
    else:
        X = remove_mostlynan_features(X)
        # TODO: Think whether we need this, was uncommented previously
        X = remove_constant_features(X)

    return X

def fast_spearman(X: pd.DataFrame) -> pd.DataFrame:
    # 1) Rank in pandas to match tie handling exactly
    R = X.rank(method="average", na_option="keep")
    A = R.to_numpy(float)
    p = A.shape[1]
    # A = R.to_numpy(dtype=np.float32) # Could be float32 for less memory, but float64 is more accurate
    C = _pearson_pairwise_nan(A)           # numba-accelerated pairwise corr
    return pd.DataFrame(C, index=X.columns, columns=X.columns)

@njit(parallel=True, fastmath=False)
def _pearson_pairwise_nan(A):
    n, p = A.shape
    out = np.empty((p, p), dtype=np.float64)

    # diagonals first
    for i in prange(p):
        out[i, i] = 1.0

    # upper triangle
    for i in prange(p):
        xi = A[:, i]
        for j in range(i + 1, p):
            x = xi
            y = A[:, j]

            # pairwise mask (ignore NaNs)
            m = (~np.isnan(x)) & (~np.isnan(y))
            cnt = np.sum(m)
            if cnt < 2:
                out[i, j] = np.nan
                continue

            xm = np.mean(x[m])
            ym = np.mean(y[m])
            dx = x[m] - xm
            dy = y[m] - ym
            num = np.sum(dx * dy)
            den = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
            out[i, j] = num / den if den > 0 else np.nan

    # mirror to lower triangle
    for i in prange(p):
        for j in range(i):
            out[i, j] = out[j, i]

    return out

def drop_high_corr(corr: pd.DataFrame, thr: float = 0.9):
    ac = corr.abs().copy()
    np.fill_diagonal(ac.values, 0.0)
    upper = ac.where(np.triu(np.ones(ac.shape), k=1).astype(bool))
    return upper.gt(thr).any(axis=0)[lambda s: s].index.tolist()


def advanced_filter_base_set(X: pd.DataFrame, corr_thresh: int=0.95) -> list:
    spearman_corr = fast_spearman(X)
    np.fill_diagonal(spearman_corr.values, 0)
    drop_cols = drop_high_corr(spearman_corr, thr=corr_thresh)
    return [col for col in X.columns if col not in drop_cols]

