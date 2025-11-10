import numpy as np
import pandas as pd
from numba import njit, prange
import polars as pl

from tabprep.preprocessors.type_change import CatAsNumTransformer

'''
Further filtering ideas:
- based on target_corr - if its the same, the feature is likely to contain the same info
'''


def remove_mostlynan_features(X):
    return X.loc[:, X.isna().mean() < 0.99]
def remove_constant_features(X):
    return X.loc[:, X.astype("float64").std() > 0] # float64 to avoid overflow warning

def remove_mostlynan_features_pl(X):
    means = pl.from_pandas(X.isna()).select(pl.all().mean()).to_pandas().iloc[0]
    return X.loc[:,means < 0.99]
def remove_constant_features_pl(X):
    stds = pl.from_pandas(X).select(pl.all().std(ddof=1)).to_pandas().iloc[0]
    return X.loc[:, stds > 0]

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

def _to_pl_df(obj, prefix=None):
    if isinstance(obj, pl.DataFrame):
        df = obj
    elif isinstance(obj, pd.DataFrame):
        df = pl.from_pandas(obj)
    elif isinstance(obj, np.ndarray):
        cols = [f"{prefix or 'c'}{i}" for i in range(obj.shape[1])]
        df = pl.DataFrame(obj, schema=cols)
    else:
        raise TypeError("X and X_int must be polars.DataFrame, pandas.DataFrame, or numpy.ndarray")
    return df

def compute_cross_corr(
    X, X_int,
    *,
    method: str = "spearman",       # "pearson" or "spearman"
    ddof: int = 1,
    propagate_nans: bool = False,
    spearman_rank: str = "average",
    long: bool = False             # return tidy long format if True
) -> pl.DataFrame:
    # Coerce to Polars
    X_pl     = _to_pl_df(X, prefix="x")
    Xint_pl  = _to_pl_df(X_int, prefix="y")

    # Basic checks
    if X_pl.height != Xint_pl.height:
        raise ValueError(f"Row mismatch: X has {X_pl.height} rows, X_int has {Xint_pl.height} rows.")
    if not X_pl.columns or not Xint_pl.columns:
        return pl.DataFrame()

    # Keep only numeric columns (preserve order)
    X_pl    = X_pl.select(pl.col(pl.NUMERIC_DTYPES))
    Xint_pl = Xint_pl.select(pl.col(pl.NUMERIC_DTYPES))
    left, right = X_pl.columns, Xint_pl.columns
    if not left or not right:
        return pl.DataFrame()

    # Combine horizontally once for ranking / correlation
    both = pl.concat([X_pl, Xint_pl], how="horizontal")

    # Spearman: rank all columns once, then use Pearson on the ranks
    if method.lower() == "spearman":
        both = both.select(pl.all().rank(method=spearman_rank))
        corr_method = "pearson"
    else:
        corr_method = "pearson"

    # Build all pairwise correlations (|left| * |right| scalars) in one vectorized select
    exprs = [
        pl.corr(pl.col(a), pl.col(b), method=corr_method, ddof=ddof, propagate_nans=propagate_nans)
        .alias(f"{a}|{b}")
        for a in left for b in right
    ]
    flat = both.select(exprs)  # single-row DataFrame

    # Reshape into matrix or long form
    tidy = (
        flat.melt(variable_name="pair", value_name="rho")
        .with_columns(pl.col("pair").str.split_exact("|", 1).alias("split"))
        .unnest("split")
        .rename({"field_0": "feature_left", "field_1": "feature_right"})
    )

    if long:
        # feature_left x feature_right x rho (tidy)
        # preserve the original column order for readability
        return tidy.with_columns(
            pl.col("feature_left").cast(pl.Categorical).rank("dense").alias("_lr")
        ).sort(["_lr", "feature_right"]).drop("_lr")

    # Wide m×n matrix: rows = X’s columns, cols = X_int’s columns
    mat = (
        tidy.pivot(values="rho", index="feature_left", columns="feature_right", aggregate_function="first")
        .sort("feature_left")
        .select(["feature_left", *right])   # enforce exact column order of X_int
        .rename({"feature_left": ""})        # optional: blank row label column
    )
    return mat

import polars as pl
import polars.selectors as cs

def spearman_corrwith_multi_pl(A: pl.DataFrame, B: pl.DataFrame, index_name: str = "") -> pl.DataFrame:
    """
    Spearman correlations between each numeric column of A (rows) and each numeric column of B (columns).
    Preserves the original column names and their order for both A and B.

    Returns a wide matrix with the first column holding A's column names (named `index_name`)
    and the remaining columns exactly B's original column names, in original order.
    """
    # numeric-only, preserving original order
    A = A.select(cs.numeric())
    B = B.select(cs.numeric())
    colsA = A.columns
    colsB = B.columns
    if not colsA or not colsB:
        return pl.DataFrame({index_name: colsA}) if colsA else pl.DataFrame()

    # Spearman = Pearson on ranks
    rA = A.select(pl.all().rank(method="average"))
    rB = B.select(pl.all().rank(method="average"))

    # all pairwise correlations in one query
    exprs = [
        pl.corr(pl.col(a), pl.col(b), method="pearson").alias(f"{a}|{b}")
        for a in colsA for b in colsB
    ]
    flat = rA.hstack(rB).select(exprs)

    # reshape to matrix and ENFORCE original orders for rows and columns
    mat = (
        flat.melt(variable_name="pair", value_name="rho")
        .with_columns(pl.col("pair").str.split_exact("|", 1).alias("split"))
        .unnest("split")
        .rename({"field_0": "__row__", "field_1": "__col__"})
        .pivot(values="rho", index="__row__", columns="__col__", aggregate_function="first")
    )

    # enforce original row order via a join key, and original column order via select
    order = pl.DataFrame({"__row__": colsA, "__ord__": range(len(colsA))})
    mat = (
        mat.join(order, on="__row__", how="left")
        .sort("__ord__")
        .drop("__ord__")
        .rename({"__row__": index_name})
        .select([index_name, *colsB])  # keeps B's original column names & order
    )
    return mat

import polars as pl
import polars.selectors as cs

def spearman_AB_chunked_pl(A: pl.DataFrame, B: pl.DataFrame, index_name: str = "column", block_size: int = 128) -> pl.DataFrame:
    """
    Compute Spearman correlations between each numeric column of A (rows, typically fewer)
    and each numeric column of B (columns, typically more), in chunks of B's columns.

    Parameters
    ----------
    A : pl.DataFrame
        Smaller dataframe (fewer columns). Each column becomes one row in the output.
    B : pl.DataFrame
        Larger dataframe (more columns). Columns become output columns.
    index_name : str, default="column"
        Name of the row-label column (A's column names).
    block_size : int, default=128
        Number of B columns to process per chunk.

    Returns
    -------
    pl.DataFrame
        Wide correlation matrix (rows = A columns, columns = B columns).
    """
    # --- numeric-only, preserve order ---
    A = A.select(cs.numeric())
    B = B.select(cs.numeric())
    colsA, colsB = A.columns, B.columns
    if not colsA or not colsB:
        return pl.DataFrame({index_name: colsA}) if colsA else pl.DataFrame()

    # --- rank once (Spearman = Pearson on ranks) ---
    rankedA = A.select(pl.all().rank(method="average"))
    rankedB = B.select(pl.all().rank(method="average"))

    # --- prebuild row order ---
    out = pl.DataFrame({index_name: colsA})

    # --- chunk loop ---
    for i in range(0, len(colsB), block_size):
        Bblock = colsB[i : i + block_size]

        # join ranks for A + this chunk of B
        tmp = rankedA.hstack(rankedB.select(Bblock))

        # build correlation expressions for all pairs in this block
        exprs = [
            pl.corr(pl.col(a), pl.col(b), method="pearson").alias(f"{a}|{b}")
            for a in colsA for b in Bblock
        ]
        flat = tmp.select(exprs)

        # reshape into tidy A×Bblock submatrix
        mat = (
            flat.melt(variable_name="pair", value_name="rho")
            .with_columns(pl.col("pair").str.split_exact("|", 1).alias("split"))
            .unnest("split")
            .rename({"field_0": "__row__", "field_1": "__col__"})
            .pivot(values="rho", index="__row__", columns="__col__", aggregate_function="first")
        )

        # enforce A's row order, keep this block’s columns only
        order = pl.DataFrame({"__row__": colsA, "__ord__": range(len(colsA))})
        mat = (
            mat.join(order, on="__row__", how="right")
               .sort("__ord__")
               .drop("__ord__")
               .rename({"__row__": index_name})
               .select([index_name, *Bblock])
        )

        # concatenate horizontally to the output
        out = out.join(mat, on=index_name, how="left")

    return out

def filter_cross_correlation(X, X_int, corr_threshold=0.95, block_size=1000):
    # First optimized (Test on 5-order diabetes: 42.777ss)
    # cross_corr = compute_cross_corr(X=X, X_int=X_int, method="spearman").to_pandas().set_index('')
    # After a few hours of experiments and vibe coding (Test on 5-order diabetes: 24.139s)
    # cross_corr = spearman_corrwith_multi_pl(pl.from_pandas(X), pl.from_pandas(X_int)).to_pandas().set_index('')
    # Chunked version
    cross_corr = spearman_AB_chunked_pl(pl.from_pandas(X), pl.from_pandas(X_int), index_name="", block_size=block_size).to_pandas().set_index("")
    drop_cols = cross_corr.columns[np.any(cross_corr.abs()>corr_threshold,axis=0).values].to_list()
    return [col for col in X_int.columns if col not in drop_cols]

def spearman_corr_matrix_pl(df: pl.DataFrame) -> pl.DataFrame:
    num = df.select(pl.col(pl.NUMERIC_DTYPES))
    cols = num.columns
    ranked = num.select(pl.all().rank(method="average"))
    exprs = [
        pl.corr(pl.col(a), pl.col(b), method="pearson").alias(f"{a}|{b}")
        for a in cols for b in cols
    ]
    flat = ranked.select(exprs)
    mat = (
        flat.melt(variable_name="pair", value_name="rho")
        .with_columns(pl.col("pair").str.split_exact("|", 1).alias("split"))
        .unnest("split")
        .rename({"field_0": "row", "field_1": "col"})
        .pivot(values="rho", index="row", columns="col", aggregate_function="first")
        .sort("row")
        .select(["row", *cols])
        .rename({"row": ""})
    )
    return mat

def filter_spearman_pl(X_int: pl.DataFrame, corr_threshold=0.95, verbose=True):
    X_int_pl = pl.from_pandas(X_int)
    cr = spearman_corr_matrix_pl(X_int_pl).to_pandas().set_index('')
    use_cols = advanced_filter_base_set(X_int, corr_thresh=corr_threshold)

    if verbose:
        print(f"No. of div features: {len([i for i in use_cols if '_/_' in i])}")
        print(f"No. of prod features: {len([i for i in use_cols if '_*_' in i])}")
        print(f"No. of add features: {len([i for i in use_cols if '_+' in i])}")
        print(f"No. of sub features: {len([i for i in use_cols if '_-_' in i])}")

    return use_cols

from typing import Literal, Optional
import numpy as np
import polars as pl

# ---------- Conversion ----------

def pandas_to_polars_numeric(df_pd) -> pl.DataFrame:
    num_pd = df_pd.select_dtypes(include="number")
    try:
        import pyarrow as pa
        tbl = pa.Table.from_pandas(num_pd, preserve_index=False)
        return pl.from_arrow(tbl)
    except Exception:
        return pl.from_pandas(num_pd)

# ---------- Fast Spearman (vectorized) ----------

def spearman_corr_pl_from_pandas(
    df_pd,
    na: Literal["drop_rows", "mean_rank"] = "mean_rank",
    batch_rows: Optional[int] = None,
) -> pl.DataFrame:
    """
    Compute Spearman correlation (Pearson on ranks) and return a Polars DataFrame.
    Returns a matrix with a first column "" listing feature names (like your original).
    """
    df_pl = pandas_to_polars_numeric(df_pd)
    cols = df_pl.columns
    if not cols:
        return pl.DataFrame({"": []})

    # Rank-transform in Polars
    if na == "drop_rows":
        ranked = df_pl.drop_nulls().select(pl.all().rank(method="average"))
    else:
        ranked = (
            df_pl.select(pl.all().rank(method="average"))
            .with_columns([pl.col(c).fill_null(pl.col(c).mean()) for c in cols])
        )

    if ranked.height < 2:
        corr_df = pl.from_numpy(np.eye(len(cols)), schema=cols)
        return pl.DataFrame({"": cols}).hstack(corr_df)

    # ---- Compute Pearson on ranks ----
    if batch_rows:  # streaming accumulation over rows
        p = len(cols)
        sum_xy = np.zeros((p, p), dtype=np.float64)
        sum_x  = np.zeros(p, dtype=np.float64)
        sum_x2 = np.zeros(p, dtype=np.float64)
        n_total = 0

        for start in range(0, ranked.height, batch_rows):
            # FIX: don't pass dtype positionally to to_numpy()
            chunk = np.asarray(ranked.slice(start, batch_rows).to_numpy(), dtype=np.float64)
            n = chunk.shape[0]
            if n == 0:
                continue
            n_total += n
            sum_xy += chunk.T @ chunk
            sum_x  += chunk.sum(axis=0)
            sum_x2 += (chunk**2).sum(axis=0)

        mean = sum_x / n_total
        var  = sum_x2 / n_total - mean**2
        std  = np.sqrt(np.maximum(var, 0.0))
        std[std == 0.0] = 1.0  # avoid div-by-zero

        cov  = sum_xy / n_total - np.outer(mean, mean)
        corr = cov / np.outer(std, std)
    else:  # all-at-once
        # FIX: don't pass dtype positionally to to_numpy()
        arr = np.asarray(ranked.to_numpy(), dtype=np.float64)
        arr -= arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, ddof=0, keepdims=True)
        std[std == 0.0] = 1.0
        arr /= std
        corr = (arr.T @ arr) / (arr.shape[0] - 1)

    # Use from_numpy with schema for column names
    corr_df = pl.from_numpy(corr, schema=cols)
    return pl.DataFrame({"": cols}).hstack(corr_df)

# ---------- Your filter wrapper (pandas in, Polars inside) ----------

def filter_spearman_pl_from_pandas(
    X_int_pd,
    corr_threshold: float = 0.95,
    na: Literal["drop_rows", "mean_rank"] = "mean_rank",
    batch_rows: Optional[int] = None,
    verbose: bool = True,
    return_corr_pl: bool = False,
):
    cr = spearman_corr_pl_from_pandas(
        X_int_pd, na=na, batch_rows=batch_rows
    )
    cr = cr.to_pandas().set_index('')

    # advanced_filter_base_set unchanged (expects pandas)
    # use_cols = advanced_filter_base_set(X_int_pd, corr_thresh=corr_threshold)
    drop_cols = drop_high_corr(cr, thr=corr_threshold)
    use_cols =  [col for col in X_int_pd.columns if col not in drop_cols]


    # if verbose:
    #     print(f"No. of div features: {sum('_/_' in c for c in use_cols)}")
    #     print(f"No. of prod features: {sum('_*_' in c for c in use_cols)}")
    #     print(f"No. of add features: {sum('_+'  in c for c in use_cols)}")
    #     print(f"No. of sub features: {sum('_-_' in c for c in use_cols)}")

    if return_corr_pl:
        return use_cols#, cr_pl
    return use_cols

def get_target_corr(X: pd.DataFrame, y: pd.Series, top_k: int = 5, use_polars=False) -> pd.Series:
    if not use_polars:
        return X.corrwith(y, method='spearman').abs().sort_values(ascending=False).iloc[:top_k]

    # Keep only numeric columns
    X_num = X.select_dtypes(include=[np.number])
    if X_num.empty:
        return pd.Series(dtype=float, name="spearman")

    # Convert to Polars
    X_pl = pl.from_pandas(X_num)
    y_pl = pl.Series(y.values)

    # Rank-transform (Spearman = Pearson on ranks)
    X_rank = X_pl.select(pl.all().rank("average"))
    y_rank = y_pl.rank("average")

    # Compute correlations
    corrs = X_rank.select([pl.corr(pl.col(c), y_rank, method="pearson").alias(c) for c in X_rank.columns])
    s = pd.Series(corrs.row(0), index=X_rank.columns, name="spearman")

    return s.abs().sort_values(ascending=False).head(top_k)
