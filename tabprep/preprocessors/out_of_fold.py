import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from typing import Literal

from tabprep.preprocessors.base import BasePreprocessor

import numpy as np
from typing import Callable, Union
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

WeightsT = Union[str, Callable[[np.ndarray], np.ndarray]]

def _stable_select(dist: np.ndarray, idx: np.ndarray, k: int):
    """
    Deterministically sort neighbors by (distance, index) and
    return the top-k *with distances aligned*.
    """
    n, K = idx.shape
    k = min(k, K)
    out_idx = np.empty((n, k), dtype=idx.dtype)
    out_dist = np.empty((n, k), dtype=dist.dtype)
    for i in range(n):
        order = np.lexsort((idx[i], dist[i]))[:k]
        out_idx[i] = idx[i, order]
        out_dist[i] = dist[i, order]
    return out_dist, out_idx

def _weighted_mean_targets(y_train: np.ndarray, idx: np.ndarray, dist: np.ndarray, weights: WeightsT):
    """
    Compute row-wise weighted mean of y_train at idx using dist-based weights.
    - weights='uniform'  -> simple mean
    - weights='distance' -> inverse distance; if any zero-distance, average zeros only
    - weights=callable   -> w = weights(dist) with shape like dist
    """
    if isinstance(weights, str):
        if weights == 'uniform':
            return y_train[idx].mean(axis=1)
        elif weights == 'distance':
            # handle zero distances: if present, average only zero-dist neighbors
            zero_mask = (dist == 0)
            out = np.empty(dist.shape[0], dtype=float)
            for i in range(dist.shape[0]):
                if zero_mask[i].any():
                    out[i] = y_train[idx[i, zero_mask[i]]].mean()
                else:
                    w = 1.0 / dist[i]
                    w_sum = w.sum()
                    out[i] = np.dot(w, y_train[idx[i]]) / w_sum if w_sum > 0 else y_train[idx[i]].mean()
            return out
        else:
            raise ValueError("weights must be 'uniform', 'distance', or a callable")
    else:
        # callable
        w = weights(dist)  # expect same shape as dist
        if w.shape != dist.shape:
            raise ValueError("weights(dist) must return an array of same shape as dist")
        # avoid division by zero issues; user-supplied callable owns semantics
        w_sum = w.sum(axis=1)
        # fallback to uniform if any row has zero total weight
        fallback = (w_sum == 0)
        out = (y_train[idx] * w).sum(axis=1, where=~np.isnan(w)) / np.where(w_sum == 0, 1, w_sum)
        if fallback.any():
            out[fallback] = y_train[idx[fallback]].mean(axis=1)
        return out

class OOFKNNTargetMeanEncoder(BasePreprocessor):
    def __init__(
        self,
        n_neighbors=5,
        metric='minkowski',
        p=2,
        exclude_self_on_train_transform=True,
        scaler: Literal['standard', 'quantile'] = 'standard',
        weights: WeightsT = 'uniform',   # <── NEW
        **kwargs
    ):
        super().__init__(keep_original=True)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.exclude_self_on_train_transform = exclude_self_on_train_transform
        self.weights = weights  # <── NEW
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=kwargs.get("random_state", 42))

    def _make_nn(self, X):
        return NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 so we can drop self in LOO
            algorithm='brute',
            metric=self.metric,
            p=self.p,
            n_jobs=1
        ).fit(X)

    def _fit(self, X_in, y_in=None, **kwargs):
        X = check_array(X_in.copy(), accept_sparse=False)
        y = np.asarray(y_in, dtype=float)
        X = self.scaler.fit_transform(X)
        self._X_train_ = X
        self._y_train_ = y
        self._nn_full_ = self._make_nn(self._X_train_)

        # LOO train feature: query once, drop self, then aggregate with weights
        dist, idx = self._nn_full_.kneighbors(self._X_train_)
        # stable order, then drop self and keep next k
        dist, idx = _stable_select(dist, idx, self.n_neighbors + 1)
        dist, idx = dist[:, 1:self.n_neighbors+1], idx[:, 1:self.n_neighbors+1]
        te = _weighted_mean_targets(self._y_train_, idx, dist, self.weights)
        self._oof_feature_ = te.reshape(-1, 1)
        return self

    def _transform(self, X, is_train=False):
        check_is_fitted(self, ['_nn_full_', '_X_train_', '_y_train_', '_oof_feature_'])
        X_idx = X.index
        Xnum = check_array(X, accept_sparse=False)
        Xnum = self.scaler.transform(Xnum)

        dist, idx = self._nn_full_.kneighbors(Xnum)
        # stable order first
        dist, idx = _stable_select(dist, idx, self.n_neighbors + 1)

        # If X is exactly the original train, drop self; else take first k
        if Xnum.shape == self._X_train_.shape and np.allclose(Xnum, self._X_train_):
            dist, idx = dist[:, 1:self.n_neighbors+1], idx[:, 1:self.n_neighbors+1]
        else:
            dist, idx = dist[:, :self.n_neighbors], idx[:, :self.n_neighbors]

        feat = _weighted_mean_targets(self._y_train_, idx, dist, self.weights)
        return pd.DataFrame(feat.reshape(-1, 1),
                            columns=[f"knn_target_mean_k{self.n_neighbors}"],
                            index=X_idx)










# def _stable_neighbors(dist, idx, k):
#     """
#     Deterministically break ties by (distance, index).
#     dist, idx: arrays from kneighbors (shape: [n_samples, k])
#     Returns top-k indices after stable tie-breaking.
#     """
#     n, K = idx.shape
#     if K == k:
#         # Still enforce tie stability (kneighbors is distance-sorted already)
#         out = np.empty_like(idx)
#         for i in range(n):
#             order = np.lexsort((idx[i], dist[i]))  # secondary key: index
#             out[i] = idx[i, order]
#         return out
#     else:
#         out = np.empty((n, k), dtype=idx.dtype)
#         for i in range(n):
#             order = np.lexsort((idx[i], dist[i]))[:k]
#             out[i] = idx[i, order]
#         return out

# class OOFKNNTargetMeanEncoder(BasePreprocessor):
#     """
#     Deterministic out-of-fold kNN target mean feature.
#     - fit/fit_transform: 5-fold OOF (default)
#     - transform: uses a single prefit NN on full train (no refit)
#     - stable tie-breaking & single-thread brute-force for determinism
#     """
#     def __init__(
#         self,
#         n_neighbors=5,
#         n_splits=5,
#         shuffle=True,
#         random_state=42,
#         metric='minkowski',
#         p=2,
#         exclude_self_on_train_transform=True,
#         scaler: Literal['standard', 'quantile'] = 'standard',
#         **kwargs
#     ):
#         super().__init__(keep_original=True)
#         self.n_neighbors = n_neighbors
#         self.n_splits = n_splits
#         self.shuffle = shuffle
#         self.random_state = random_state
#         self.metric = metric
#         self.p = p
#         self.exclude_self_on_train_transform = exclude_self_on_train_transform

#         if scaler == 'standard':
#             self.scaler = StandardScaler()
#         elif scaler == 'quantile':
#             self.scaler = QuantileTransformer(output_distribution='normal', random_state=random_state)

#     def _make_nn(self, X):
#         # algorithm='brute' + n_jobs=1 for deterministic behavior
#         return NearestNeighbors(
#             n_neighbors=self.n_neighbors + 1,  # +1 to allow self-exclusion later
#             algorithm='brute',
#             metric=self.metric,
#             p=self.p,
#             n_jobs=1
#         ).fit(X)

#     # def _fit(self, X_in, y_in=None, **kwargs):
#     #     X = X_in.copy()
#     #     y = y_in.copy()
        
#     #     X = check_array(X, accept_sparse=False)
#     #     y = np.asarray(y, dtype=float)


#     #     X = self.scaler.fit_transform(X)
#     #     self._X_train_ = X
#     #     self._y_train_ = y
#     #     self._nn_full_ = self._make_nn(self._X_train_)

#     #     # OOF feature (deterministic CV)
#     #     kf = KFold(
#     #         n_splits=self.n_splits,
#     #         shuffle=self.shuffle,
#     #         random_state=self.random_state if self.shuffle else None
#     #     )
#     #     oof = np.zeros(X.shape[0], dtype=float)

#     #     for tr_idx, val_idx in kf.split(X):
#     #         nn = self._make_nn(X[tr_idx])
#     #         dist, idx = nn.kneighbors(X[val_idx])
#     #         # keep exactly n_neighbors neighbors (no self in OOF because val_idx not in tr_idx)
#     #         idx = _stable_neighbors(dist, idx, self.n_neighbors)
#     #         oof[val_idx] = y[tr_idx][idx].mean(axis=1)

#     #     self._oof_feature_ = oof.reshape(-1, 1)
#     #     return self

#     def _fit(self, X_in, y_in=None, **kwargs):
#         X = X_in.copy()
#         y = y_in.copy()
        
#         X = check_array(X, accept_sparse=False)
#         y = np.asarray(y, dtype=float)

#         X = self.scaler.fit_transform(X)
#         self._X_train_ = X
#         self._y_train_ = y
#         self._nn_full_ = self._make_nn(self._X_train_)

#         # *** LOO version ***
#         # compute LOO train feature: query with X, then drop self from neighbor set
#         dist, idx = self._nn_full_.kneighbors(X)

#         # drop first neighbour (self) then take next k
#         idx = _stable_neighbors(dist, idx, self.n_neighbors + 1)
#         idx = idx[:, 1:self.n_neighbors+1]

#         self._oof_feature_ = y[idx].mean(axis=1).reshape(-1,1)
#         return self


#     # def _fit_transform(self, X, y=None, **fit_params):
#     #     self.fit(X, y)
#     #     return pd.DataFrame(self._oof_feature_, columns=[f"knn_target_mean_k{self.n_neighbors}"])

#     # def _transform(self, X, is_train=False):
#     #     check_is_fitted(self, ['_nn_full_', '_X_train_', '_y_train_', '_oof_feature_'])
#     #     if is_train:
#     #         return pd.DataFrame(self._oof_feature_, columns=[f"knn_target_mean_k{self.n_neighbors}"], index=X.index)
#     #     X_idx = X.index
#     #     X = check_array(X, accept_sparse=False)
#     #     X = self.scaler.transform(X)
#     #     # Use full-train neighbors (deterministic)
#     #     dist, idx = self._nn_full_.kneighbors(X)

#     #     # If user passes the original training matrix, optionally drop self-neighbor
#     #     if self.exclude_self_on_train_transform and X.shape == self._X_train_.shape and np.allclose(X, self._X_train_):
#     #         # For exact same matrix, self is guaranteed to be a neighbor at distance 0; drop it.
#     #         keep_k = self.n_neighbors
#     #         # Sort ties deterministically first
#     #         idx = _stable_neighbors(dist, idx, self.n_neighbors + 1)
#     #         idx = idx[:, 1:keep_k+1]
#     #     else:
#     #         # Deterministic tie-breaking to exactly n_neighbors
#     #         idx = _stable_neighbors(dist, idx, self.n_neighbors)

#     #     feat = self._y_train_[idx].mean(axis=1)
#     #     return pd.DataFrame(feat.reshape(-1, 1), columns=[f"knn_target_mean_k{self.n_neighbors}"], index=X_idx)

#     def _transform(self, X, is_train=False):
#         check_is_fitted(self, ['_nn_full_', '_X_train_', '_y_train_', '_oof_feature_'])
#         X_idx = X.index
#         X = check_array(X, accept_sparse=False)
#         X = self.scaler.transform(X)

#         dist, idx = self._nn_full_.kneighbors(X)
#         idx = _stable_neighbors(dist, idx, self.n_neighbors + 1)

#         # if this is the *exact* original train -> drop self row
#         if X.shape == self._X_train_.shape and np.allclose(X, self._X_train_):
#             idx = idx[:, 1:self.n_neighbors+1]
#         else:
#             idx = idx[:, :self.n_neighbors]

#         feat = self._y_train_[idx].mean(axis=1)
#         return pd.DataFrame(feat.reshape(-1,1),
#                             columns=[f"knn_target_mean_k{self.n_neighbors}"],
#                             index=X_idx)


#     def get_feature_names_out(self, input_features=None):
#         return np.array([f"knn_target_mean_k{self.n_neighbors}"])
