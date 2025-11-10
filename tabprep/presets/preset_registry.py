import pandas as pd

from typing import Dict, Any, Optional, List
import copy

from tabprep.presets.lgb_presets import get_lgb_presets

class LGBPresetRegistry:
    """
    Registry for LightGBM hyperparameter presets.

    Primary methods:
      - available_presets(): list all preset names
      - get_params(target_type, preset, overrides=None): get a full param dict

    Notes:
      - Presets are returned as-is and can be overridden via `overrides`.
      - Extend `_default_presets()` to add/change built-ins.
    """

    def __init__(self, 
                 target_type: str, 
                 presets: Optional[Dict[str, Dict[str, Any]]] = None,
                 use_experimental: bool = False,
                 ) -> None:
        self._presets: Dict[str, Dict[str, Any]] = self.get_default_presets()
        if presets:
            # user-supplied presets override defaults
            self._presets.update(presets)
        if use_experimental:
            from tabprep.presets.lgb_presets_experimental import get_experimental_presets
            self._presets.update(get_experimental_presets())

        for preset_name, preset in self._presets.items():
            self._presets[preset_name] = self.map_preprocessors(preset, target_type)

    def map_preprocessors(self, preset: dict, target_type: str) -> list:
        from tabprep.preprocessors.preprocessor_map import get_preprocessor
        if len(preset['prep_params']) == 0:
            return preset
        
        preprocessors = []
        for prep_name, init_params in preset['prep_params'].items():
            preprocessor_class = get_preprocessor(prep_name)
            if preprocessor_class is not None:
                preprocessors.append(preprocessor_class(target_type=target_type, **init_params))
            else:
                raise ValueError(f"Preprocessor {prep_name} not recognized.")
        preset['prep_params'] = preprocessors
        return preset

    def available_presets(self) -> List[str]:
        """Return available preset names."""
        return sorted(self._presets.keys())

    def get_params(
        self,
        target_type: str,
        preset: str = 'default',
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a LightGBM param dict for a given `target_type` and `preset`.

        Args:
            target_type: LightGBM objective, e.g. 'binary', 'multiclass', 'regression'.
            preset: name of the preset in the registry.
            overrides: optional dict of param overrides merged last.

        Returns:
            A dict suitable for initializing a LightGBM model.

        Raises:
            ValueError: if the preset name is unknown.
        """
        if preset not in self._presets:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {', '.join(self.available_presets())}"
            )

        # Base scaffold; extend as you add adaptivity and preprocessing knobs.
        init_params: Dict[str, Any] = {
            "objective": target_type,
            "boosting_type": "gbdt",
            "n_estimators": 10000,
            "verbosity": -1,
            **{key: value for key, value in self._presets[preset].items() if key not in  ["prep_params", "cv_params"]}
        }

        # init_params.update(self._presets[preset]['init_params'])
        # if overrides:
        #     init_params.update(overrides)

        if 'cv_params' in self._presets[preset]:
            cv_params = self._presets[preset]['cv_params']
        else:
            cv_params = dict()

        return init_params, self._presets[preset]['prep_params'], cv_params

    def presets_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return a deep copy of the full preset dictionary."""
        return copy.deepcopy(self._presets)

    def filter_presets(self, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       target_type: str
                       ) -> None:
        # TODO: Add dataset adaptivity

        '''
        Aspects: 
        a) The presets own filter requirements as str; 
        b) Filter functions based on that strings; 
        c) dataset objects

        Process:
        1. Gather relevant dataset statistics: 
        - Feature types (categorical, numerical)
        - Missing value patterns
        - Cardinality of categorical features
        - Distribution of numerical features

        2. Filter presets:


        '''

        pass

    def combine_presets(self, presets: List[Dict]):
        # TODO: Preset combination and reduction techniques
        pass

    @staticmethod
    def get_default_presets() -> Dict[str, Dict[str, Any]]:
        # TODO: Add risk ranking to presets
        # TODO: Remove experimental presets
        return get_lgb_presets()

    def update_default_presets(self, target_type: str, new_presets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        presets = self._default_presets(target_type=target_type)
        presets.update(new_presets)
        return presets
    

