import numpy as np
import pandas as pd
from tabprep.preprocessors.base import NumericBasePreprocessor
from scipy.stats import spearmanr, rankdata
import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import rankdata
import itertools

class ArithmeticBySpearmanPreprocessor(NumericBasePreprocessor):
    def __init__(self, operations=None, correlation_threshold=0.05, max_depth=2, top_k=None, base_k=20, target_type="auto"):
        """
        operations: list of arithmetic operations to apply ["add", "sub", "mul", "div"]
        correlation_threshold: minimum improvement in Spearman correlation required to keep a new feature
        max_depth: maximum number of successive operations to combine features
        top_k: optional, keep only top_k features per depth based on correlation
        base_k: optional, number of most correlated original features to keep as base (default 20)
        target_type: one of {"auto", "regression", "binary", "multiclass"}
        """
        super().__init__(keep_original=True)
        if operations is None:
            operations = ["+", "-", "*", "/"]
        self.operations = operations
        self.correlation_threshold = correlation_threshold
        self.max_depth = max_depth
        self.top_k = top_k
        self.base_k = base_k
        self.target_type = target_type

        self.selected_features_ = []
        self.feature_formulas_ = []  # store operations for reproducibility on unseen data
        self.baseline_corr_ = {}

    def _apply_operation(self, f1, f2, op):
        if op == "+":
            return f1 + f2
        elif op == "-":
            return f1 - f2
        elif op == "*":
            return f1 * f2
        elif op == "/":
            return np.where(f2 != 0, f1 / f2, 0)
        else:
            raise ValueError(f"Unsupported operation: {op}")

    def _spearman_corr_vectorized(self, X_rank, y_rank):
        """Compute Spearman correlation between each column in X_rank and y_rank using Pearson on ranks."""
        X_rank = np.asarray(X_rank)
        y_rank = np.asarray(y_rank)

        X_mean = X_rank.mean(axis=0)
        y_mean = y_rank.mean()

        num = np.sum((X_rank - X_mean) * (y_rank[:, None] - y_mean), axis=0)
        den = np.sqrt(np.sum((X_rank - X_mean) ** 2, axis=0) * np.sum((y_rank - y_mean) ** 2))
        corr = num / den
        return corr

    def _generate_combinations(self, features, y_rank, baseline_corr, depth, prefix=None):
        new_features = []
        feature_names = list(features.keys())

        for f1, f2 in itertools.combinations(feature_names, 2):
            for op in self.operations:
                new_feature = self._apply_operation(features[f1], features[f2], op)
                new_feature_rank = rankdata(new_feature)

                corr = self._spearman_corr_vectorized(new_feature_rank.reshape(-1, 1), y_rank)[0]
                if np.isnan(corr):
                    continue

                if abs(corr) > max(baseline_corr.get(f1, 0), baseline_corr.get(f2, 0)) + self.correlation_threshold:
                    name = f"({f1}_{op}_{f2})"
                    if prefix:
                        name = f"{prefix}_{name}"
                    if name not in features:
                        new_features.append((name, new_feature, abs(corr), (f1, f2, op)))

        # prune to top_k if specified
        if self.top_k and len(new_features) > self.top_k:
            new_features = sorted(new_features, key=lambda x: x[2], reverse=True)[:self.top_k]

        return new_features

    def _fit_regression_or_binary(self, X, y):
        y_rank = rankdata(y)

        # baseline correlations (vectorized)
        X_rank = np.apply_along_axis(rankdata, 0, X.values)
        corr = self._spearman_corr_vectorized(X_rank, y_rank)

        baseline_corr = {col: abs(c) for col, c in zip(X.columns, corr)}
        self.baseline_corr_ = baseline_corr

        # Select only top base_k features
        top_base_features = sorted(baseline_corr.items(), key=lambda x: x[1], reverse=True)[: self.base_k]
        base_feature_names = [f for f, _ in top_base_features]

        # Initialize with top base features
        features = {col: X[col].values for col in base_feature_names}

        for depth in range(1, self.max_depth + 1):
            new_features = self._generate_combinations(features, y_rank, baseline_corr, depth)
            for name, f, corr, formula in new_features:
                features[name] = f
                baseline_corr[name] = corr
                self.selected_features_.append((name, f))
                self.feature_formulas_.append((name, formula))

    def _fit_multiclass(self, X, y):
        for cls in y.unique():
            y_binary = (y == cls).astype(int)
            y_rank = rankdata(y_binary)

            # baseline correlations (vectorized)
            X_rank = np.apply_along_axis(rankdata, 0, X.values)
            corr = self._spearman_corr_vectorized(X_rank, y_rank)

            baseline_corr = {col: abs(c) for col, c in zip(X.columns, corr)}

            # Select only top base_k features
            top_base_features = sorted(baseline_corr.items(), key=lambda x: x[1], reverse=True)[: self.base_k]
            base_feature_names = [f for f, _ in top_base_features]

            # Initialize with top base features
            features = {col: X[col].values for col in base_feature_names}

            for depth in range(1, self.max_depth + 1):
                new_features = self._generate_combinations(features, y_rank, baseline_corr, depth, prefix=f"class{cls}")
                for name, f, corr, formula in new_features:
                    features[name] = f
                    baseline_corr[name] = corr
                    self.selected_features_.append((name, f))
                    self.feature_formulas_.append((name, formula))

    def _fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()

        self.selected_features_ = []
        self.feature_formulas_ = []
        self.baseline_corr_ = {}

        if self.target_type == "auto":
            if y.nunique() > 2 and y.dtype.kind in "iO":
                target_type = "multiclass"
            elif y.nunique() == 2:
                target_type = "binary"
            else:
                target_type = "regression"
        else:
            target_type = self.target_type

        if target_type in ["regression", "binary"]:
            self._fit_regression_or_binary(X, y)
        elif target_type == "multiclass":
            self._fit_multiclass(X, y)
        else:
            raise ValueError(f"Unsupported target_type: {target_type}")

        return self

    def _transform(self, X_in):
        X = X_in.copy()
        X_out = pd.DataFrame(index = X.index)
        feature_cache = {col: X[col].values for col in X.columns}

        for name, (f1, f2, op) in self.feature_formulas_:
            if f1 not in feature_cache or f2 not in feature_cache:
                continue
            feature_cache[name] = self._apply_operation(feature_cache[f1], feature_cache[f2], op)
            X_out[name] = feature_cache[name]

        return X_out
