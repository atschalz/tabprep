from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from autogluon.features.generators import (
    ArithmeticFeatureGenerator,
    BulkFeatureGenerator,
    CategoricalInteractionFeatureGenerator,
    GroupByFeatureGenerator,
    OOFTargetEncodingFeatureGenerator,
)

try:
    from autogluon.features.generators import RandomSubsetFeatureCompressionGenerator
except ImportError:
    # Some AutoGluon source checkouts expose RSFC only from the concrete module.
    from autogluon.features.generators.rsfc import RandomSubsetFeatureCompressionGenerator

from autogluon.features.generators.abstract import AbstractFeatureGenerator


_FEATURE_GENERATOR_CLASS_MAP = {
    "ArithmeticFeatureGenerator": ArithmeticFeatureGenerator,
    "BulkFeatureGenerator": BulkFeatureGenerator,
    "CategoricalInteractionFeatureGenerator": CategoricalInteractionFeatureGenerator,
    "GroupByFeatureGenerator": GroupByFeatureGenerator,
    "OOFTargetEncodingFeatureGenerator": OOFTargetEncodingFeatureGenerator,
    "RandomSubsetFeatureCompressionGenerator": RandomSubsetFeatureCompressionGenerator,
}


class TabPrepFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    TabPrep feature generator wrapper.

    This mirrors the nested `ag.prep_params` structure used by AutoGluon TabPrep
    configs and exposes it through a scikit-learn compatible fit/transform API.
    """

    def __init__(
        self,
        target_type: str = "binary",
        random_state: int | None = 0,
        use_groupby: bool = True,
        groupby_kwargs: dict[str, Any] | None = None,
        use_rsfc: bool = True,
        rsfc_kwargs: dict[str, Any] | None = None,
        use_arithmetic: bool = True,
        arithmetic_kwargs: dict[str, Any] | None = None,
        use_cat_interact: bool = True,
        cat_interact_kwargs: dict[str, Any] | None = None,
        use_oofte: bool = True,
        oofte_kwargs: dict[str, Any] | None = None,
    ):
        self.target_type = target_type
        self.random_state = random_state
        self.use_groupby = use_groupby
        self.groupby_kwargs = None if groupby_kwargs is None else dict(groupby_kwargs)
        self.use_rsfc = use_rsfc
        self.rsfc_kwargs = None if rsfc_kwargs is None else dict(rsfc_kwargs)
        self.use_arithmetic = use_arithmetic
        self.arithmetic_kwargs = None if arithmetic_kwargs is None else dict(arithmetic_kwargs)
        self.use_cat_interact = use_cat_interact
        self.cat_interact_kwargs = None if cat_interact_kwargs is None else dict(cat_interact_kwargs)
        self.use_oofte = use_oofte
        self.oofte_kwargs = None if oofte_kwargs is None else dict(oofte_kwargs)

        self.preprocessor_: AbstractFeatureGenerator | None = None
        self.generator_config_: dict[str, Any] | None = None
        self.feature_metadata_in_ = None
        self.feature_metadata_ = None
        self.features_in_ = None
        self.features_out_ = None
        self.is_fitted_ = False

    def _get_generator_config(self) -> dict[str, Any]:
        """
        Build the TabPrep-style generator config used by the model-side mixin.
        """

        stage: list[list[Any]] = []
        passthrough_types = None

        if self.use_groupby:
            stage.append(["GroupByFeatureGenerator", self.groupby_kwargs or {}])

        if self.use_rsfc:
            stage.append(["RandomSubsetFeatureCompressionGenerator", self.rsfc_kwargs or {}])

        if self.use_arithmetic:
            arithmetic_kwargs = dict(self.arithmetic_kwargs or {})
            arithmetic_kwargs.setdefault("passthrough", True)
            stage.append(["ArithmeticFeatureGenerator", arithmetic_kwargs])

        cat_stage: list[list[Any]] = []
        if self.use_cat_interact:
            cat_kwargs = dict(self.cat_interact_kwargs or {})
            cat_kwargs.setdefault("passthrough", True)
            cat_stage.append(["CategoricalInteractionFeatureGenerator", cat_kwargs])

        if self.use_oofte:
            cat_stage.append(["OOFTargetEncodingFeatureGenerator", self.oofte_kwargs or {}])
            passthrough_types = {"invalid_raw_types": ["category", "object"]}

        if cat_stage:
            stage.append(cat_stage)

        prep_config: dict[str, Any] = {"ag.prep_params": [stage] if stage else []}
        if passthrough_types is not None:
            prep_config["ag.prep_params.passthrough_types"] = passthrough_types

        return prep_config

    def _init_preprocessor(
        self,
        preprocessor_cls: type[AbstractFeatureGenerator] | str,
        init_params: dict | None,
    ) -> AbstractFeatureGenerator:
        if isinstance(preprocessor_cls, str):
            try:
                preprocessor_cls = _FEATURE_GENERATOR_CLASS_MAP[preprocessor_cls]
            except KeyError as err:
                raise ValueError(f"Unknown preprocessor class name: {preprocessor_cls}") from err

        init_params = {} if init_params is None else dict(init_params)
        init_params.setdefault("verbosity", 0)
        init_params.setdefault("target_type", self.target_type)
        if "random_state" not in init_params and self.random_state is not None:
            init_params["random_state"] = self.random_state

        return preprocessor_cls(**init_params)

    def _recursive_init_preprocessors(self, prep_param: tuple | list[list | tuple]):
        if isinstance(prep_param, list):
            if len(prep_param) == 0:
                param_type = "list"
            elif len(prep_param) == 2 and isinstance(prep_param[0], (str, AbstractFeatureGenerator)):
                param_type = "generator"
            else:
                param_type = "list"
        elif isinstance(prep_param, tuple):
            param_type = "generator"
        else:
            raise ValueError(f"Invalid value for prep_param: {prep_param}")

        if param_type == "list":
            out = []
            for p in prep_param:
                out.append(self._recursive_init_preprocessors(p))
            return out
        if param_type == "generator":
            assert len(prep_param) == 2
            preprocessor_cls = prep_param[0]
            init_params = prep_param[1]
            return self._init_preprocessor(preprocessor_cls=preprocessor_cls, init_params=init_params)
        raise ValueError(f"Invalid value for prep_param: {prep_param}")

    def get_preprocessor(self) -> AbstractFeatureGenerator | None:
        config = self._get_generator_config()
        self.generator_config_ = config

        prep_params = config.get("ag.prep_params", None)
        passthrough_types = config.get("ag.prep_params.passthrough_types", None)
        if not prep_params:
            return None

        # Keep the outer list intact so nested generator groups remain parallel stages.
        preprocessors = self._recursive_init_preprocessors(prep_param=prep_params)
        if len(preprocessors) == 0:
            return None
        if len(preprocessors) == 1 and isinstance(preprocessors[0], AbstractFeatureGenerator):
            return preprocessors[0]

        return BulkFeatureGenerator(
            generators=preprocessors,
            remove_unused_features="false_recursive",
            post_drop_duplicates=True,
            passthrough=True,
            passthrough_types=passthrough_types,
            verbosity=0,
        )

    def _validate_fit_input(self, X: pd.DataFrame, y: pd.Series | None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        if y is None:
            raise AssertionError("y must be provided to fit TabPrepFeatureGenerator")

    def _fit_preprocessor(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.preprocessor_ = self.get_preprocessor()
        if self.preprocessor_ is None:
            return self._reorder_output_columns(X.copy())

        X_out = self.preprocessor_.fit_transform(X, y)
        self.feature_metadata_in_ = getattr(self.preprocessor_, "feature_metadata_in", None)
        self.feature_metadata_ = getattr(self.preprocessor_, "feature_metadata", None)
        self.features_in_ = getattr(self.preprocessor_, "features_in", None)
        self.features_out_ = getattr(self.preprocessor_, "features_out", None)
        return self._reorder_output_columns(X_out)

    def _reorder_output_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Put original numeric input columns first while preserving the rest of the
        generated feature order.
        """

        numeric_cols = [col for col in getattr(self, "numeric_features_in_", []) if col in X.columns]
        remaining_cols = [col for col in X.columns if col not in numeric_cols]
        if not numeric_cols:
            return X
        return X.loc[:, numeric_cols + remaining_cols]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._validate_fit_input(X, y)
        X = X.copy()
        y = pd.Series(y).copy()

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        self.numeric_features_in_ = list(X.select_dtypes(include="number").columns)
        self._fit_preprocessor(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise AttributeError("TabPrepFeatureGenerator must be fit before calling transform")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        if self.preprocessor_ is None:
            return self._reorder_output_columns(X.copy())
        return self._reorder_output_columns(self.preprocessor_.transform(X))

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        self._validate_fit_input(X, y)
        X = X.copy()
        y = pd.Series(y).copy()

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        self.numeric_features_in_ = list(X.select_dtypes(include="number").columns)
        X_out = self._fit_preprocessor(X, y)
        self.is_fitted_ = True
        return X_out
