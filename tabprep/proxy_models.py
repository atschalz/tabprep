import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from scipy.special import expit
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import PolynomialFeatures

from typing import Literal, List

class TargetMeanClassifier(ClassifierMixin, BaseEstimator):
    """Vectorized target-mean classifier with NaN-as-own-category support."""
    def __init__(self):
        self._estimator_type = "classifier"
        # unique object sentinel for NaNs
        self._nan_sentinel = object()

    def fit(self, X, y):
        # 1) Seriesify and sentinelize NaNs
        X_ser = self._to_series(X)
        # X_filled = X_ser.where(X_ser.notna(), self._nan_sentinel)

        # 2) factorize X into integer codes (0…n_categories-1)
        codes, uniques = pd.factorize(X_ser, sort=False)
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
        # X_filled = X_ser.where(X_ser.notna(), self._nan_sentinel)

        # vectorized lookup: returns –1 for unseen categories
        codes = self._categories_index.get_indexer(X_ser)
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
        self.mapping_ = df.groupby('feature', observed=False)['target'].mean().to_dict()
        self.mapping_ = {k: v for k, v in self.mapping_.items() if pd.notna(v)}
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
        return preds

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

class TargetMeanClassifierCut(TargetMeanClassifier):
    def __init__(self, q_thresh=0):
        super().__init__()
        self.q_thresh = q_thresh

    def fit(self, X, y):
        X_use = X.iloc[:,0].copy()
        self.c_map = dict(X_use.value_counts())
        self.c_map = {k: k  if v > self.q_thresh else 'nan' for k, v in self.c_map.items()}  
        X_use = X_use.map(self.c_map)
        return super().fit(X_use.to_frame(), y)

    def predict_proba(self, X):
        X_use = X.iloc[:,0].copy()
        X_use = X_use.map(self.c_map)
        return super().predict_proba(X_use.to_frame())

class TargetMeanRegressorCut(TargetMeanRegressor):
    def __init__(self, q_thresh=0):
        super().__init__()
        self.q_thresh = q_thresh

    def fit(self, X, y):
        # TODO: Make sure that the category dtype of train and val matches
        X_use = X.iloc[:,0].copy()
        self.c_map = dict(X_use.value_counts())
        self.c_map = {k: k  if v > self.q_thresh else 'nan' for k, v in self.c_map.items()}  
        X_use = X_use.map(self.c_map)
        return super().fit(X_use.to_frame(), y)

    def predict(self, X):
        X_use = X.iloc[:,0].copy()
        X_use = X_use.map(self.c_map)
        return super().predict(X_use.to_frame())

class LightGBMClassifierCut(LGBMClassifier):
    def __init__(self, q_thresh=0, init_kwargs: dict=dict()):
        super().__init__(**init_kwargs)
        self.verbose = init_kwargs.get("verbose", -1)
        self.n_estimators = init_kwargs.get("n_estimators", 100)
        self.max_bin = init_kwargs.get("max_bin", 255)
        self.max_depth = init_kwargs.get("max_depth", 2)
        self.random_state = init_kwargs.get("random_state", 42)
        self.init_kwargs = None
        self.q_thresh = q_thresh

    def fit(self, X, y, **kwargs):
        X_use = X.iloc[:,0].copy()
        if X_use.dtype in ['object', 'category']:
            fill = 'nan'
        else:
            fill = np.nan
        self.c_map = dict(X_use.value_counts())
        self.c_map = {k: k  if v > self.q_thresh else fill for k, v in self.c_map.items()}  # only keep those with more than 5 occurrences
        X_use = X_use.map(self.c_map)
        if X_use.dtype in ['object', 'category']:
            X_use = X_use.astype('category')
            self.dt = X_use.dtype 
        return super().fit(X_use.to_frame(), y, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        X_use = X.iloc[:,0].copy()
        X_use = X_use.map(self.c_map)
        if X_use.dtype in ['object', 'category']:   
            X_use = X_use.astype(self.dt)
        return super().predict_proba(X_use.to_frame(), **kwargs)

class LightGBMRegressorCut(LGBMRegressor):
    def __init__(self, q_thresh=0, init_kwargs: dict=dict()):
        super().__init__(**init_kwargs)
        self.verbose = init_kwargs.get("verbose", -1)
        self.n_estimators = init_kwargs.get("n_estimators", 100)
        self.max_bin = init_kwargs.get("max_bin", 255)
        self.max_depth = init_kwargs.get("max_depth", 2)
        self.random_state = init_kwargs.get("random_state", 42)
        self.init_kwargs = None
        self.q_thresh = q_thresh

    def fit(self, X, y, **kwargs):
        X_use = X.iloc[:,0].copy()
        if X_use.dtype in ['object', 'category']:
            fill = 'nan'
        else:
            fill = np.nan
        self.c_map = dict(X_use.value_counts())
        self.c_map = {k: k  if v > self.q_thresh else fill for k, v in self.c_map.items()}  # only keep those with more than 5 occurrences
        X_use = X_use.map(self.c_map)
        if X_use.dtype in ['object', 'category']:
            X_use = X_use.astype('category')
            self.dt = X_use.dtype 
        return super().fit(X_use.to_frame(), y, **kwargs)

    def predict(self, X, **kwargs):
        X_use = X.iloc[:,0].copy()
        X_use = X_use.map(self.c_map)
        if X_use.dtype in ['object', 'category']:   
            X_use = X_use.astype(self.dt)
        return super().predict(X_use.to_frame(), **kwargs)

class LightGBMBinner(BaseEstimator, TransformerMixin):
    """
    Hybrid LightGBM‐style binning:
      1) Try quantile‐sketch with linear interpolation.
      2) If that yields NaN or <=1 cut, fallback to uniform slicing of unique values.
    Guarantees up to max_bin bins for any continuous feature.
    """
    def __init__(self, max_bin=255, subsample_for_bin=200000):
        self.max_bin = max_bin
        self.subsample_for_bin = subsample_for_bin
        self.bin_thresholds_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        # capture feature names if DataFrame
        self.feature_names_in_ = (
            getattr(X, "columns", None)
            if hasattr(X, "columns")
            else [f"f{i}" for i in range(n_features)]
        )
        thresholds = []
        for j in range(n_features):
            col = X[:, j]
            # 1) quantile‐sketch
            #   subsample if needed
            if n_samples > self.subsample_for_bin:
                idx = np.random.choice(n_samples, self.subsample_for_bin, replace=False)
                col_s = np.sort(col[idx])
            else:
                col_s = np.sort(col)
            # compute interior quantiles with linear interpolation
            qs = np.linspace(0, 1, self.max_bin + 1)[1:-1]
            cuts = np.quantile(col_s, qs, method="linear")
            cuts = np.unique(cuts[~np.isnan(cuts)])
            # 2) fallback if bad
            if len(cuts) <= 1:
                uni = np.unique(col)
                m = len(uni)
                if m <= self.max_bin:
                    cuts = uni
                else:
                    # pick evenly‐spaced unique indices
                    idx2 = np.linspace(1, m - 1, self.max_bin + 1, dtype=int)[1:-1]
                    cuts = uni[idx2]
            thresholds.append(cuts.tolist())
        self.bin_thresholds_ = thresholds
        return self

    def transform(self, X):
        if self.bin_thresholds_ is None:
            raise RuntimeError("Must fit before transform")
        X = np.asarray(X)
        B = np.zeros_like(X, dtype=int)
        for j, thr in enumerate(self.bin_thresholds_):
            B[:, j] = np.digitize(X[:, j], thr, right=True)
        return B

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_, dtype=object)

class KMeansBinner(BaseEstimator, TransformerMixin):
    """
    1D k-means–based binning:
      - For each feature, cluster its (subsampled) values into up to max_bin clusters.
      - Sort the cluster centers, then place thresholds at midpoints between them.
      - Transform by np.digitize against those thresholds.

    Buckets will not be equal‐frequency, but they will adapt to your data's density.
    """
    def __init__(self, max_bin=255, subsample_for_bin=100000, random_state=None):
        self.max_bin = max_bin
        self.subsample_for_bin = subsample_for_bin
        self.random_state = random_state
        self.bin_thresholds_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # capture feature names if DataFrame, else generic
        self.feature_names_in_ = (
            getattr(X, "columns", None)
            if hasattr(X, "columns")
            else [f"f{i}" for i in range(n_features)]
        )

        thresh_list = []
        for j in range(n_features):
            col = X[:, j]

            # subsample for clustering
            if n_samples > self.subsample_for_bin:
                idx = np.random.RandomState(self.random_state).choice(
                    n_samples, self.subsample_for_bin, replace=False
                )
                sample = col[idx].reshape(-1, 1)
            else:
                sample = col.reshape(-1, 1)

            # decide k = min(max_bin, unique_count)
            unique_vals = np.unique(sample.ravel())
            n_clusters = min(self.max_bin, unique_vals.shape[0])

            # fit 1D k-means
            km = MiniBatchKMeans(n_clusters=n_clusters,
                                 random_state=self.random_state)
            km.fit(sample)
            centers = np.sort(km.cluster_centers_.ravel())

            # thresholds = midpoints between sorted centers
            mids = (centers[:-1] + centers[1:]) / 2.0
            thresh_list.append(mids.tolist())

        self.bin_thresholds_ = thresh_list
        return self

    def transform(self, X):
        if self.bin_thresholds_ is None:
            raise RuntimeError("Must fit before calling transform()")
        X = np.asarray(X)
        n_samples, n_features = X.shape
        Xb = np.zeros_like(X, dtype=int)

        for j, thr in enumerate(self.bin_thresholds_):
            Xb[:, j] = np.digitize(X[:, j], thr, right=True)

        return Xb

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_, dtype=object)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from skrub import SquashingScaler

class CustomModel(BaseEstimator):
    def __init__(self, 
                 target_type,
                 scaler='squashing',
                 standardize_target=True,
                 cat_method:Literal['ohe','oof-te']='ohe',
                 scale_binary:bool=True,
                 **kwargs
                 ):

        self.target_type = target_type
        self.standardize_target = standardize_target
        self.cat_method = cat_method
        self.scale_binary = scale_binary
        
        if self.target_type == 'regression' and self.standardize_target:
            self.target_scaler = StandardScaler()
        elif self.target_type in ['binary', 'multiclass']:
            self.target_scaler = None
        else:
            raise ValueError("target_type must be 'binary', 'multiclass', or 'regression'")

        self.pipeline = None
        self.model: BaseEstimator = None
        if self.cat_method == 'oof-te':
            from tabprep.preprocessors.categorical import OOFTargetEncoder
            self.oof_te = OOFTargetEncoder(target_type=self.target_type, keep_original=False, alpha=10.)
        else:
            self.oof_te = None

        # --------------- feature scaling ------------ #
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'quantile-normal':
            from sklearn.preprocessing import QuantileTransformer
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif scaler == 'quantile-uniform':
            from sklearn.preprocessing import QuantileTransformer
            self.scaler = QuantileTransformer(output_distribution='uniform')
        elif scaler == 'squashing':
            self.scaler = SquashingScaler()
        elif scaler == 'squashing':
            self.scaler = SquashingScaler()
        elif scaler is None:
            self.scaler = 'passthrough'
        else:
            raise ValueError("scaler must be 'standard', 'quantile-normal', 'quantile-uniform', 'squashing', or None")

    def _fit_preprocessor(self, X, y, **kwargs):
        if self.cat_method == 'oof-te':
            X = self.oof_te.fit_transform(X, y)

        if not self.scale_binary:
            X.loc[:, X.nunique() == 2] = X.loc[:, X.nunique() == 2].astype('object')

        # Determine which columns are categorical or numerical
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        # Define transformers for preprocessing
        transformers = [
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', self.scaler)
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=100))
                # ('pass', 'passthrough')
            ]), categorical_cols)
        ]

        # Combine transformations in a ColumnTransformer
        self.pipeline = ColumnTransformer(transformers)

        # Fit the transformers and transform the data
        self.pipeline.fit(X)

        if self.target_type == 'regression':
            self.target_scaler.fit(y.values.reshape(-1, 1))


    def preprocess(self, X, y=None, is_train=False, **kwargs):
        if self.cat_method == 'oof-te':
            X = self.oof_te.transform(X, is_train=is_train)

        if not self.scale_binary:
            X.loc[:, X.nunique() == 2] = X.loc[:, X.nunique() == 2].astype('object')

        X = self.pipeline.transform(X)
        if self.target_type == 'regression' and y is not None:
            y = self.target_scaler.transform(y.values.reshape(-1, 1)).flatten()
        return X, y

    def fit(self, X_in, y_in, **kwargs):
        X = X_in.copy()
        y = y_in.copy()
        
        self._fit_preprocessor(X, y)            
        X, y = self.preprocess(X, y, is_train=True) 
        
        self.model.fit(X, y)

        return self

    def predict(self, X, is_train=False, **kwargs):
        # Transform the features using the fitted pipeline
        X_transformed, _ = self.preprocess(X, is_train=is_train)

        # Predict based on the model type
        if self.target_type == 'regression':
            y_pred_scaled = self.model.predict(X_transformed)
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            return y_pred
        elif self.target_type == 'binary':
            return self.model.predict_proba(X_transformed)[:, 1]
        elif self.target_type == 'multiclass':
            return self.model.predict_proba(X_transformed)
        else:
            raise ValueError("target_type must be 'binary', 'regression' or 'classification'")
        
    def decision_function(self, X, is_train=False, **kwargs):
        # FIXME: Think what to do for regression
        # Transform the features using the fitted pipeline
        if self.cat_method == 'oof-te':
            X = self.oof_te.transform(X, is_train=is_train)
        X_transformed = self.pipeline.transform(X)

        if self.target_type == 'binary':
            return self.model.decision_function(X_transformed)
        elif self.target_type == 'multiclass':
            return self.model.decision_function(X_transformed)
        else:
            raise ValueError("target_type must be 'binary' or 'classification'")

from sklearn.model_selection import KFold, StratifiedKFold
class OOFCustomModel:
    def __init__(self, 
                 target_type,
                 base_model_cls: BaseEstimator,
                 base_model_kwargs: None,
                 n_splits:int=5,
                 random_state:int=42,
                 ):
        assert target_type in {"regression","binary","multiclass"}
        self.target_type = target_type
        self.n_splits = n_splits
        self.random_state = random_state

        if base_model_kwargs is None:
            base_model_kwargs = {}

        if self.target_type == "regression":
            self.kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            self.kf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)

        self.fold_models_: List[BaseEstimator] = [base_model_cls(**base_model_kwargs) for _ in range(self.n_splits)]
        self.full_model: BaseEstimator = base_model_cls(**base_model_kwargs)

    # -----------------------------------------------------------
    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()

        oof_preds = []
        for num, (tr, val) in enumerate(self.kf.split(X, y)):
            X_tr = X.iloc[tr]
            y_tr = y.iloc[tr]   
            X_val = X.iloc[val]

            self.fold_models_[num].fit(X_tr, y_tr)
            if self.target_type == "multiclass":
                oof_preds.append(pd.DataFrame(self.fold_models_[num].predict(X_val), index=X.index[val]))
            else:
                oof_preds.append(pd.Series(self.fold_models_[num].predict(X_val), index=X.index[val]))

        self.train_preds_ = pd.concat(oof_preds, axis=0).loc[X.index]
        self.train_decisions_ = pd.concat(oof_preds, axis=0).loc[X.index].values

        self.full_model.fit(X, y)

        return self

    # -----------------------------------------------------------
    def predict(self, X_in, is_train:bool=False, **kwargs):
        X = X_in.copy()

        # TODO: Make sure that the order of samples in X matches training data. 
        if is_train:
            # return stored OOF train encodings
            # rather than recomputing
            assert hasattr(self,"train_preds_"), "fit() not called"
            return self.train_preds_.copy()

        return self.full_model.predict(X, is_train=is_train, **kwargs)
    
    def decision_function(self, X_in, is_train:bool=False, **kwargs):
        X = X_in.copy()

        # TODO: Make sure that the order of samples in X matches training data. 
        if is_train:
            # return stored OOF train encodings
            # rather than recomputing
            assert hasattr(self,"train_decisions_"), "fit() not called"
            return self.train_decisions_.copy()

        return self.full_model.decision_function(X, is_train=is_train, **kwargs)

class CustomLinearModel(CustomModel):
    def __init__(self, 
                 target_type,
                 scaler='squashing',
                 standardize_target=True,
                 cat_method='ohe',
                 scale_binary=True,
                 linear_model_type='lasso',    # linear, lasso, ridge
                 lambda_: Literal['low', 'medium', 'high', float] = 'medium', # single regularization strength param
                 max_degree: int = 1
                 ):
        super().__init__(target_type=target_type, scaler=scaler, standardize_target=standardize_target, cat_method=cat_method, scale_binary=scale_binary)

        self.linear_model_type = linear_model_type
        self.max_degree = max_degree

        if isinstance(lambda_, str):
            self.lambda_ = self.set_lambda_from_category(cat=lambda_)
        else:
            self.lambda_ = lambda_

        if self.target_type == 'regression':
            if self.linear_model_type == 'linear':
                self.model = LinearRegression()
            elif self.linear_model_type == 'lasso':
                self.model = Lasso(alpha=self.lambda_)   # λ ↑ ⇒ more reg
            elif self.linear_model_type == 'ridge':
                self.model = Ridge(alpha=self.lambda_)   # λ ↑ ⇒ more reg
            else:
                raise ValueError("linear_model_type must be 'linear', 'lasso', or 'ridge'")

        # --------------- classification ------------- #
        elif self.target_type in ['binary', 'multiclass']:

            # linear => no regularization
            if self.linear_model_type == 'linear':
                penalty = 'l2'
                C = 1e9           # effectively infinite C = no reg
                solver = 'lbfgs'

            else:
                # map lasso/ridge
                penalty = 'l1' if self.linear_model_type == 'lasso' else 'l2'

                # λ direction must match regression:
                # λ ↑ ⇒ stronger regularization
                # if self.lambda_ <= 0:
                #     C = 1e9
                # else:
                #     C = 1.0 / self.lambda_

                # auto solver
                solver = 'saga' if penalty == 'l1' else 'lbfgs'

            self.model = LogisticRegression(
                penalty=penalty,
                C=self.lambda_,
                solver=solver,
                max_iter=2000,
                # multi_class='auto'
            )
            self.target_scaler = None
        else:
            raise ValueError("target_type must be 'binary', 'multiclass', or 'regression'")
        
        if self.max_degree > 1:
            self.model = Pipeline([
                ("poly", PolynomialFeatures(degree=self.max_degree, include_bias=False)),
                ("ridge", self.model)
            ])

    def set_lambda_from_category(self, cat: str) -> float:
        """
        Set self.lambda_ to a single value chosen for the given category
        ('low' | 'medium' | 'high') for this (target_type, linear_model_type).
        Values are picked on a log scale from realistic ranges.
        """
        if cat not in {"low", "medium", "high"}:
            raise ValueError("cat must be one of {'low','medium','high'}")

        table = {
            "regression": {
                "linear":   {"low": 0.0,      "medium": 0.0,     "high": 0.0},
                "ridge":    {"low": 0.1,   "medium": 1.,  "high": 10.},
                # "lasso":    {"low": 6.3e-6,   "medium": 1.0e-4,  "high": 1.6e-3},
                "lasso":    {"low": 1e-2,   "medium": 1e-3,  "high": 1e-4},
            },
            "binary": {
                # 'linear' maps to L1 in your setup
                "linear":   {"low": 4.0e-6,   "medium": 3.2e-5,  "high": 2.5e-4},
                "ridge":    {"low": 10.,   "medium": 1.,  "high": 0.1},
                "lasso":    {"low": 10.,   "medium": 1,  "high": 0.1},
            },
            "multiclass": {
                # slightly stronger than binary for stability
                "linear":   {"low": 1.0e-5,   "medium": 1.0e-4,  "high": 7.5e-4},
                "ridge":    {"low": 2.0e-4,   "medium": 3.0e-3,  "high": 5.0e-2},
                # "lasso":    {"low": 1.0e-5,   "medium": 1.0e-4,  "high": 7.5e-4},
                "lasso":    {"low": 10,   "medium": 1,  "high": 0.1},
            },
        }

        try:
            lambda_ = table[self.target_type][self.linear_model_type][cat]
        except KeyError:
            raise ValueError(f"Unsupported combination target_type={self.target_type!r}, linear_model_type={self.linear_model_type!r}")
        return lambda_

class OOFCustomLinearModel(OOFCustomModel):
    def __init__(self, 
                 target_type,
                 scaler='squashing',
                 standardize_target=True,
                 cat_method='ohe',
                 scale_binary=True,
                 n_splits:int=5,
                 random_state:int=42,
                 linear_model_type='lasso',    # linear, lasso, ridge
                 lambda_: Literal['low', 'medium', 'high', float] = 'medium',                    # single regularization strength param
                 max_degree: int = 1
                 ):
        base_model_kwargs = {
            'target_type': target_type,
            'scaler': scaler,
            'standardize_target': standardize_target,
            'cat_method': cat_method,
            'scale_binary': scale_binary,
            'linear_model_type': linear_model_type,
            'lambda_': lambda_,
            'max_degree': max_degree,
        }
        super().__init__(
            target_type=target_type,
            base_model_cls=CustomLinearModel,
            base_model_kwargs=base_model_kwargs,
            n_splits=n_splits,
            random_state=random_state
        )

class GroupedCustomLinearModel:
    def __init__(self, target_type, group_col='auto', min_samples=200):
        self.target_type = target_type
        self.group_col   = group_col
        self.min_samples = min_samples

    def fit(self, X, y):
        # --- global model ---
        self.global_model_ = CustomLinearModel(target_type=self.target_type)
        self.global_model_.fit(X, y)

        if self.group_col == 'auto':
            self.group_col = X.select_dtypes(include=['object', 'category']).nunique().idxmax()

        # --- group models ---
        self.group_models_ = {}
        for g in X[self.group_col].unique():
            X_use = X[X[self.group_col]==g]
            y_use = y[X[self.group_col]==g]
            print(f'Category {g}: {len(X_use)} samples')
            if len(X_use)>20:
                self.group_models_[g] = CustomLinearModel(target_type=self.target_type).fit(X_use, y_use)
            else:
                # not enough data for group-specific model
                self.group_models_[g] = self.global_model_

        return self

    def predict(self, X):
        # baseline = global model
        out = pd.Series(
            self.global_model_.predict(X),
            index=X.index
        )

        # override on groups with enough data
        for g, m in self.group_models_.items():
            idx = (X[self.group_col] == g)
            if idx.any():
                out.loc[idx] = m.predict(X.loc[idx])

        return out
    
    def decision_function(self, X):
        # baseline = global model
        out = pd.Series(
            self.global_model_.decision_function(X),
            index=X.index
        )

        # override on groups with enough data
        for g, m in self.group_models_.items():
            idx = (X[self.group_col] == g)
            if idx.any():
                out.loc[idx] = m.decision_function(X.loc[idx])

        return out

from scipy.special import logit
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
class CustomKNNModel(CustomModel):
    def __init__(self, 
                 target_type,
                 scaler='squashing',
                 standardize_target=True,
                 n_neighbors:int=20,
                 weights:str='distance',
                 p:float=2,
                 ):
        super().__init__(target_type=target_type, scaler=scaler, standardize_target=standardize_target)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        # FIXME: Currently we never use the kNN model directly without the OOF model. If we would want to do so, we would need logic that excludes train self-samples to avoid data leakage.

        if self.target_type == 'regression':
            self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)
        elif self.target_type == 'binary':
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)
        elif self.target_type == 'multiclass':
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)
        else:
            raise ValueError("target_type must be 'binary', 'multiclass', or 'regression'")

        import numpy as np
        from scipy.special import logit

    def decision_function(self, X, eps=1e-15):
        X_transformed = self.pipeline.transform(X)
        p = np.clip(self.model.predict_proba(X_transformed), eps, 1-eps)

        if self.target_type == "binary":
            return logit(p[:, 1])

        # multiclass (OvR)
        return logit(p)

class OOFCustomKNNModel(OOFCustomModel):
    def __init__(self, 
                 target_type,
                 scaler='squashing',
                 standardize_target=True,
                 n_splits:int=5,
                 random_state:int=42,
                 n_neighbors:int=20,
                 weights:str='distance',
                 p:float=2,
                 ):
        base_model_kwargs = {
            'target_type': target_type,
            'scaler': scaler,
            'standardize_target': standardize_target,
            'n_neighbors': n_neighbors,
            'weights': weights,
            'p': p
        }
        super().__init__(
            target_type=target_type,
            base_model_cls=CustomKNNModel,
            base_model_kwargs=base_model_kwargs,
            n_splits=n_splits,
            random_state=random_state
        )

# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# class CustomKNNModel(BaseEstimator):
#     def __init__(self, target_type, standardize_target=True, n_neighbors=1, weights='uniform'):
#         self.target_type = target_type  # 'regression' or 'classification'
#         self.standardize_target = standardize_target
#         self.n_neighbors = n_neighbors
#         self.weights = weights

#         # Initialize internal variables
#         self.model = None
#         self.target_scaler = None
#         self.pipeline = None

#     def fit(self, X_in, y_in):
#         X = X_in.copy()
#         y = y_in.copy()
#         # Determine which columns are categorical or numerical
#         categorical_cols = X.select_dtypes(include=['object', 'category']).columns
#         numerical_cols = X.select_dtypes(include=[np.number]).columns

#         # Define transformers for preprocessing
#         transformers = [
#             ('num', Pipeline([
#                 ('imputer', SimpleImputer(strategy='mean')),
#                 ('scaler', StandardScaler())
#             ]), numerical_cols),
#             ('cat', Pipeline([
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=100))
#             ]), categorical_cols)
#         ]

#         # Combine transformations in a ColumnTransformer
#         self.pipeline = ColumnTransformer(transformers)

#         # Fit the transformers and transform the data
#         self.pipeline.fit(X)
#         X_transformed = self.pipeline.transform(X) 
        
#         if self.target_type == 'regression':
#             # Standardize the target if regression
#             self.target_scaler = StandardScaler()
#             y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

#             # Fit the regression model (using KNeighborsRegressor)
#             self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)
#             self.model.fit(X_transformed, y_scaled)
#         elif self.target_type in ['binary', 'multiclass']:
#             # Fit the classification model (using KNeighborsClassifier)
#             self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
#             self.model.fit(X_transformed, y)
#         else:
#             raise ValueError("target_type must be 'regression' or 'classification'")

#         return self

#     def predict(self, X):
#         # Transform the features using the fitted pipeline
#         X_transformed = self.pipeline.transform(X)

#         # Predict based on the model type
#         if self.target_type == 'regression':
#             y_pred_scaled = self.model.predict(X_transformed)
#             y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
#             return y_pred
#         elif self.target_type == 'binary':
#             return self.model.predict_proba(X_transformed)[:, 1]
#         elif self.target_type == 'multiclass':
#             return self.model.predict_proba(X_transformed)
#         else:
#             raise ValueError("target_type must be 'regression' or 'classification'")


class OOFModePredictor(OOFCustomModel):
    def __init__(self, 
                 target_type,
                 mode_value,
                 base_model_cls: BaseEstimator,
                 base_model_kwargs: dict = None,
                 n_splits:int=5,
                 random_state:int=42,
                 ):
        self.mode_value = mode_value
        if base_model_kwargs is None:
            base_model_kwargs = {'target_type': target_type}
        if 'target_type' not in base_model_kwargs:
            base_model_kwargs['target_type'] = target_type
        super().__init__(
            target_type=target_type,
            base_model_cls=base_model_cls,
            base_model_kwargs=base_model_kwargs,
            n_splits=n_splits,
            random_state=random_state
        )
        # if self.model_type == 'extra_tree' and self.target_type == 'regression':
        #     self.model = ExtraTreeRegressor(min_samples_leaf=5, random_state=self.random_state)
        # elif self.model_type == 'extra_tree' and self.target_type in ['binary', 'multiclass']:
        #     self.model = ExtraTreeClassifier(min_samples_leaf=5, random_state=self.random_state)
        # elif self.model_type == 'linear' and self.target_type == 'regression':
        #     self.model = Lasso(random_state=self.random_state)
        # elif self.model_type == 'linear' and self.target_type in ['binary', 'multiclass']:
        #     self.model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        # else:
        #     raise ValueError(f"Invalid model_type: {self.model_type}")
        
    def fit(self, X_in, y_in):
        # mode = y_in.value_counts().index[0]
        # for mode in y.value_counts().index[:10]:
        #     y_bin = (y == mode).astype(int)
        #     y_test_bin = (y_test == mode).astype(int)

        y_bin = (y_in == self.mode_value).astype(int)
        return super().fit(X_in, y_bin)
    
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor

class CustomExtraTreeModel(CustomModel):
    def __init__(self, 
                 target_type,
                 min_samples_leaf=5,
                 ):
        super().__init__(target_type=target_type)
        self.min_samples_leaf = min_samples_leaf

        if target_type == 'regression':
            self.model = ExtraTreeRegressor(min_samples_leaf=self.min_samples_leaf)
        elif target_type in ['binary', 'multiclass']:
            self.model = ExtraTreeClassifier(min_samples_leaf=self.min_samples_leaf)
        else:
            raise ValueError("target_type must be 'binary', 'multiclass', or 'regression'")

    # def fit(self, X, y):
    #     self.model.fit(X, y)
    #     return self

    # def predict(self, X):
    #     if self.target_type in ['binary', 'multiclass']:
    #         return self.model.predict_proba(X)
    #     return self.model.predict(X)

    # def decision_function(self, X):
    #     if self.target_type == 'binary':
    #         proba = self.model.predict_proba(X)
    #         return logit(proba[:, 1])
    #     elif self.target_type == 'multiclass':
    #         proba = self.model.predict_proba(X)
    #         return logit(proba)
    #     else:
    #         raise ValueError("decision_function is only available for 'binary' or 'multiclass' target types")


### Init score functionality

class OOFLinearInitScore(OOFCustomLinearModel):
    def __init__(self, target_type: str, init_kwargs=dict(), **lin_kwargs):
        super().__init__(target_type=target_type, **init_kwargs, **lin_kwargs)

    def init_score(self, X_in, is_train=False, **kwargs):
        X = X_in.copy()

        if self.target_type == "regression":
            return self.predict(X, is_train=is_train)
        else:
            return self.decision_function(X, is_train=is_train)
        
class OOFKNNInitScore(OOFCustomKNNModel):
    def __init__(self, target_type: str, init_kwargs=dict(), **lin_kwargs):
        super().__init__(target_type=target_type, **init_kwargs, **lin_kwargs)

    def init_score(self, X_in, is_train=False, **kwargs):
        X = X_in.copy()

        if self.target_type == "regression":
            return self.predict(X, is_train=is_train)
        else:
            return self.decision_function(X, is_train=is_train)

class LinearInitScore(CustomLinearModel):
    def __init__(self, target_type: str, init_kwargs=dict(), **lin_kwargs):
        super().__init__(target_type=target_type, **init_kwargs, **lin_kwargs)

    def init_score(self, X_in, **kwargs):
        X = X_in.copy()

        if self.model is None:
            raise RuntimeError("call .fit(...) first")

        if self.target_type == "regression":
            return self.predict(X)
        else:
            return self.decision_function(X)

class GroupedLinearInitScore(GroupedCustomLinearModel):
    def __init__(self, target_type: str, **lin_kwargs):
        super().__init__(target_type=target_type, **lin_kwargs)
    
    def init_score(self, X_in, **kwargs):
        X = X_in.copy()

        if self.target_type == "regression":
            return self.predict(X)

        raw = self.decision_function(X)
        return raw
