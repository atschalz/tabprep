import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from typing import Dict, Any, Iterable, Optional

class DatasetSummary:
    # TODO: Add regression distribution type statistics (e.g. skew, kurtosis, multimodality)
    """
    Summarize a dataset X and target y with:
      - global_stats: pd.Series of dataset-level single-value metrics
      - feature_stats: pd.DataFrame with features as rows and statistics as columns

    Key design choices:
      • Binary feature rule: a column is binary if it has exactly 2 non-null distinct values,
        OR it has exactly 1 non-null distinct value AND has missing values.
      • Binary features are EXCLUDED from numeric/categorical counts and from pairwise feature
        correlation maxima, but they DO appear in feature_stats with is_binary=True.
      • Global stats contain only scalars for ease of logging/export.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series | pd.DataFrame  (if DataFrame, the first column is used)
    high_cat_cardinality_threshold : int, default 10
        Threshold above which a categorical (non-binary) feature is considered high-cardinality.

    Attributes
    ----------
    global_stats : pd.Series
    feature_stats : pd.DataFrame
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        high_cat_cardinality_threshold: int = 10,
    ):
        self.X = X
        self.y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
        self.high_cat_cardinality_threshold = int(high_cat_cardinality_threshold)

        self.global_stats: pd.Series = pd.Series(dtype="object")
        self.feature_stats: pd.DataFrame = pd.DataFrame()
        self._compute_statistics()

    # ---------- helpers ----------
    @staticmethod
    def _is_classification_target(y: pd.Series, max_unique_numeric: int = 20) -> bool:
        """Heuristic: classification if non-numeric, or numeric with few unique values."""
        if pd.api.types.is_bool_dtype(y) or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
            return True
        if pd.api.types.is_numeric_dtype(y):
            return y.nunique(dropna=True) <= max_unique_numeric
        return False

    @staticmethod
    def _entropy(proportions: pd.Series) -> float:
        """Shannon entropy (base 2)."""
        p = proportions.values.astype(float)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum()) if p.size else np.nan

    @staticmethod
    def _to_numeric_if_binary(y: pd.Series) -> Optional[pd.Series]:
        """Return a 0/1 numeric encoding if y is binary; else None."""
        u = y.dropna().unique()
        if len(u) == 2:
            codes, _ = pd.factorize(y, sort=True)
            return pd.Series(codes, index=y.index, dtype="int64")
        return None

    # ---------- core ----------
    def _compute_statistics(self) -> None:
        X, y = self.X, self.y
        n_rows, n_cols = X.shape

        # ---- Identify binary features per spec
        nunique_nonnull = X.nunique(dropna=True)
        has_missing = X.isna().any()
        is_binary_mask = (nunique_nonnull == 2) | ((nunique_nonnull == 1) & has_missing)
        binary_cols = nunique_nonnull[is_binary_mask].index.tolist()
        n_binary = len(binary_cols)

        # Dtype groups (before exclusion)
        numeric_all = X.select_dtypes(include="number")
        categorical_like_all = X.select_dtypes(include=["object", "category", "bool"])
        datetime_all = X.select_dtypes(include=["datetime", "datetimetz"])

        # Exclude binary from numeric/categorical counts & summaries
        num_cols = numeric_all.drop(columns=[c for c in binary_cols if c in numeric_all.columns], errors="ignore")
        cat_cols = categorical_like_all.drop(columns=[c for c in binary_cols if c in categorical_like_all.columns], errors="ignore")

        # ---- Global single-valued metrics
        n_cells = n_rows * n_cols if n_rows and n_cols else 0
        missing_pct = (X.isnull().to_numpy().sum() / n_cells * 100.0) if n_cells else 0.0

        # Duplicates (% of rows)
        dup_pct_X = float((X.duplicated().sum() / n_rows) * 100.0) if n_rows else np.nan
        XY = pd.concat([X, y.rename("_target_")], axis=1)
        dup_pct_Xy = float((XY.duplicated().sum() / n_rows) * 100.0) if n_rows else np.nan

        # Memory usage (MB)
        memory_mb = float(X.memory_usage(deep=True).sum() / (1024 ** 2))

        # Numeric shape summaries across non-binary numeric features
        if num_cols.shape[1]:
            skew_s = num_cols.skew(numeric_only=True)
            kurt_s = num_cols.kurtosis(numeric_only=True)  # Fisher (normal -> 0)
            avg_numeric_skew = float(skew_s.mean())
            min_numeric_skew = float(skew_s.min())
            max_numeric_skew = float(skew_s.max())
            avg_numeric_kurt = float(kurt_s.mean())
            min_numeric_kurt = float(kurt_s.min())
            max_numeric_kurt = float(kurt_s.max())
        else:
            avg_numeric_skew = min_numeric_skew = max_numeric_skew = np.nan
            avg_numeric_kurt = min_numeric_kurt = max_numeric_kurt = np.nan

        # Max absolute pairwise correlations among NON-binary numeric features
        if num_cols.shape[1] >= 2:
            pearson = num_cols.corr(method="pearson", numeric_only=True).abs().values
            spearman = num_cols.corr(method="spearman", numeric_only=True).abs().values
            np.fill_diagonal(pearson, 0.0)
            np.fill_diagonal(spearman, 0.0)
            max_abs_pearson_corr = float(pearson.max())
            max_abs_spearman_corr = float(spearman.max())
        else:
            max_abs_pearson_corr = np.nan
            max_abs_spearman_corr = np.nan

        # Feature-to-sample ratio
        feature_to_sample_ratio = float(n_cols / n_rows) if n_rows else np.nan

        # Cardinalities (exclude binary)
        if num_cols.shape[1]:
            num_card = num_cols.nunique(dropna=True)
            max_numeric_cardinality = int(num_card.max())
        else:
            max_numeric_cardinality = np.nan

        if cat_cols.shape[1]:
            cat_card = cat_cols.nunique(dropna=True)
            max_categorical_cardinality = int(cat_card.max())
            n_cat_high = int((cat_card > self.high_cat_cardinality_threshold).sum())
        else:
            max_categorical_cardinality = np.nan
            n_cat_high = 0

        # Target overview & type awareness
        target_dtype = str(y.dtype)
        n_unique_target_values = int(y.nunique(dropna=False))
        is_classification = self._is_classification_target(y)

        # Classification-focused scalars
        counts = y.value_counts(dropna=False)
        if is_classification and counts.sum() > 0:
            props = counts / counts.sum()
            majority_prop = float(props.max()) if not props.empty else np.nan
            minority_prop = float(props.min()) if not props.empty else np.nan
            target_entropy = self._entropy(props) if props.size else np.nan
            n_classes = int(counts.size)
        else:
            majority_prop = minority_prop = target_entropy = np.nan
            n_classes = np.nan

        # Target correlations (for regression or binary targets)
        max_target_pearson = np.nan
        max_target_spearman = np.nan
        y_for_corr: Optional[pd.Series] = None

        if pd.api.types.is_numeric_dtype(y):
            y_for_corr = y  # regression target
        else:
            y_bin = self._to_numeric_if_binary(y)
            if y_bin is not None:
                y_for_corr = y_bin  # binary target encoded 0/1

        if y_for_corr is not None and numeric_all.shape[1] >= 1:
            aligned = pd.concat([numeric_all, y_for_corr.rename("_target_")], axis=1).dropna()
            if aligned.shape[0] >= 2:
                # absolute correlations of each numeric feature with target
                pc = aligned.corr(method="pearson", numeric_only=True)["_target_"].drop("_target_", errors="ignore").abs()
                sc = aligned.corr(method="spearman", numeric_only=True)["_target_"].drop("_target_", errors="ignore").abs()
                max_target_pearson = float(pc.max()) if not pc.empty else np.nan
                max_target_spearman = float(sc.max()) if not sc.empty else np.nan

        # Assemble global stats Series (SCALARS ONLY)
        self.global_stats = pd.Series(
            {
                # Core dataset shape/quality
                "n_rows": n_rows,
                "n_columns": n_cols,
                "missing_percentage": missing_pct,
                "memory_usage_mb": memory_mb,
                "feature_to_sample_ratio": feature_to_sample_ratio,
                # Feature type counts (excluding binary from numeric/categorical)
                "n_numeric_features": int(num_cols.shape[1]),
                "n_categorical_features": int(cat_cols.shape[1]),
                "n_datetime_features": int(datetime_all.shape[1]),
                "n_binary_features": int(n_binary),
                # Duplicates (% rows)
                "duplicate_rows_pct_X": dup_pct_X,
                "duplicate_rows_pct_Xy": dup_pct_Xy,
                # Numeric distribution shape
                "avg_numeric_skewness": avg_numeric_skew,
                "min_numeric_skewness": min_numeric_skew,
                "max_numeric_skewness": max_numeric_skew,
                "avg_numeric_kurtosis": avg_numeric_kurt,
                "min_numeric_kurtosis": min_numeric_kurt,
                "max_numeric_kurtosis": max_numeric_kurt,
                # Pairwise feature correlation maxima
                "max_abs_pearson_correlation": max_abs_pearson_corr,
                "max_abs_spearman_correlation": max_abs_spearman_corr,
                # Cardinalities (renamed per spec)
                "max_numeric_cardinality": max_numeric_cardinality,
                "max_categorical_cardinality": max_categorical_cardinality,
                "n_cat_features_high_cardinality": n_cat_high,
                "high_cat_cardinality_threshold": int(self.high_cat_cardinality_threshold),
                # Target overview
                "target_dtype": target_dtype,
                "n_unique_target_values": n_unique_target_values,
                # Classification-focused
                "is_classification_target": bool(is_classification),
                "n_classes": n_classes,
                "majority_class_proportion": majority_prop,
                "minority_class_proportion": minority_prop,
                "target_entropy": target_entropy,
                # Target correlations for regression/binary
                "max_target_pearson_correlation": max_target_pearson,
                "max_target_spearman_correlation": max_target_spearman,
            },
            dtype="object",
        )

        # ---- Feature-level DataFrame (one row per feature)
        records: list[Dict[str, Any]] = []
        # Precompute per-feature correlations with target (if available)
        feat_pc: Dict[str, float] = {}
        feat_sc: Dict[str, float] = {}
        if y_for_corr is not None:
            # compute per-feature correlations only using rows where both feature and y exist
            for col in numeric_all.columns:
                s = X[col]
                df_align = pd.concat([s, y_for_corr], axis=1).dropna()
                if df_align.shape[0] >= 2:
                    feat_pc[col] = float(df_align.corr(method="pearson").iloc[0, 1])
                    feat_sc[col] = float(df_align.corr(method="spearman").iloc[0, 1])
                else:
                    feat_pc[col] = np.nan
                    feat_sc[col] = np.nan

        for col in X.columns:
            s = X[col]
            rec: Dict[str, Any] = {
                "feature": col,
                "dtype": str(s.dtype),
            }
            n_miss = int(s.isna().sum())
            rec["n_missing"] = n_miss
            rec["missing_pct"] = float(n_miss / n_rows * 100) if n_rows else np.nan
            rec["cardinality"] = int(s.nunique(dropna=True))

            is_bin = col in binary_cols
            rec["is_binary"] = bool(is_bin)

            # Determine a high-level feature_type label for convenience
            if is_bin:
                feature_type = "binary"
            elif col in num_cols.columns:
                feature_type = "numeric"
            elif col in cat_cols.columns:
                feature_type = "categorical"
            elif col in datetime_all.columns:
                feature_type = "datetime"
            else:
                feature_type = "other"
            rec["feature_type"] = feature_type

            # Numeric summary for non-binary numeric
            if feature_type == "numeric":
                rec["mean"] = float(s.mean(skipna=True))
                rec["std"] = float(s.std(skipna=True))
                rec["min"] = float(s.min(skipna=True))
                rec["max"] = float(s.max(skipna=True))
                rec["skew"] = float(s.skew(skipna=True))
                rec["kurtosis"] = float(s.kurtosis(skipna=True))
                # correlations with target (if regression/binary)
                rec["pearson_to_target"] = feat_pc.get(col, np.nan)
                rec["spearman_to_target"] = feat_sc.get(col, np.nan)

            # For categorical: most frequent value and its count
            elif feature_type == "categorical":
                vc = s.value_counts(dropna=False)
                rec["top"] = (vc.index[0] if len(vc) else np.nan)
                rec["freq"] = (int(vc.iloc[0]) if len(vc) else np.nan)

            # Binary: store counts per level (compact as dict); no numeric moments
            elif feature_type == "binary":
                rec["levels"] = dict(s.value_counts(dropna=False).to_dict())

            records.append(rec)

        self.feature_stats = pd.DataFrame.from_records(records).set_index("feature")

    # ---------- public API ----------
    def update(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series | pd.DataFrame] = None,
        high_cat_cardinality_threshold: Optional[int] = None,
    ) -> None:
        """Replace X and/or y and/or threshold, then recompute statistics."""
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
        if high_cat_cardinality_threshold is not None:
            self.high_cat_cardinality_threshold = int(high_cat_cardinality_threshold)
        self._compute_statistics()

    def list_statistics(self) -> Dict[str, list[str]]:
        """Names of available stats separated by global vs feature-level columns."""
        return {
            "global_stats": list(self.global_stats.index),
            "feature_stats_columns": list(self.feature_stats.columns),
        }

    def __repr__(self) -> str:
        return (
            f"DatasetSummary(n_rows={self.global_stats.get('n_rows', None)}, "
            f"n_columns={self.global_stats.get('n_columns', None)}, "
            f"target_dtype={self.global_stats.get('target_dtype', None)}, "
            f"classification={self.global_stats.get('is_classification_target', None)}, "
            f"high_cat_cardinality_threshold={self.high_cat_cardinality_threshold})"
        )


