# from tabprep.detectors.groupby_interactions import GroupByFeatureEngineer
# from tabprep.detectors.num_interaction import NumericalInteractionDetector
from tabprep.preprocessors.frequency import FrequencyEncoder
from tabprep.preprocessors.categorical import CatIntAdder, CatGroupByAdder, OneHotPreprocessor, CatLOOTransformer, DropCatTransformer
from tabprep.preprocessors.numerical.scaling import SquashingScalerPreprocessor, KDITransformerPreprocessor, QuantileScalerPreprocessor, RobustScalerPreprocessor, StandardScalerPreprocessor
from tabprep.preprocessors.numerical.one_to_many import SplinePreprocessor, NumericalOneHotPreprocessor, LowCardinalityOneHotPreprocessor
from tabprep.preprocessors.numerical.one_to_one import TrigonometricTransformer, OptimalBinner, NearestNeighborDistanceTransformer
from tabprep.preprocessors.multivariate import FastICAPreprocessor, SVDPreprocessor, PCAPreprocessor, KernelPCAPreprocessor, SparsePCAPreprocessor, DictionaryLearningPreprocessor, FactorAnalysisPreprocessor, LDAPreprocessor, NMFPreprocessor, DuplicateCountAdder, DuplicateSampleLOOEncoder, LinearFeatureAdder, RandomFourierFeatureTransformer, SklearnRandomFourierFeatureTransformer
from tabprep.preprocessors.type_change import ToCategoricalTransformer, CatAsNumTransformer
from tabprep.preprocessors.binary import BinarySumPreprocessor, BinaryJaccardGrouper
from tabprep.preprocessors.numerical.arithmetic import ArithmeticBySpearmanPreprocessor, NumericalInteractionPreprocessor
from tabprep.preprocessors.misc import GroupbyInteractionPreprocessor
from typing import Dict, Any

def get_lgb_presets() -> Dict[str, Dict[str, Any]]:

    presets = {

    ### LIGHTGBM-specific presets

        "default": {
            'prep_params': {},

        },
        'huertas': { # For small sample sizes / few-shot learning
            "extra_trees": True,
            "num_leaves": 4,
            'learning_rate': 0.05,
            'min_data_in_leaf': 1,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 1,
            'min_data_per_group': 1,
            'cat_l2': 0,
            'cat_smooth': 0,
            'max_cat_to_onehot': 100,
            'min_data_in_bin': 3,
            'prep_params': {},
        },
        'aggressive_feature_bagging': { # Preset for extremely high-dimensional data
            'feature_fraction': 0.2,
            'prep_params': {},
        },
        # TODO: Add data-adaptive feature bagging
        'heavy_regularization': { # Preset for very noisy data
            'cat_l2': 10.,
            'cat_smooth': 10.,
            'min_data_per_group': 10,
            'max_cat_threshold': 32, 
            "max_bin": 128,  
            'min_data_in_leaf': 20,
            'min_sum_hessian_in_leaf': 1e-3,
            'lambda_l1': 3.0,
            'lambda_l2': 3.0,
            'min_gain_to_split': 0.5,
            'min_data_in_bin': 3,
            'bagging_fraction': 0.7,
            'feature_fraction': 0.7,
            'bagging_freq': 1,
            'prep_params': {},
        },
        # TODO: Make residual logic work
        # 'linear_residuals': { # Very strong linearity
        #     'init_params': {},
        #     'preprocessors': [],
        #     'cv_params': {'linear_residuals': True},
        # },
        'no_regularization': { # very strong train/val(/test) alignment
            'cat_l2': 0,
            'cat_smooth': 0,
            'min_data_per_group': 1,
            'max_cat_threshold': 100000,
            "max_bin": 10000,  # TODO: Test even larger
            'min_data_in_leaf': 1,
            'min_sum_hessian_in_leaf': 1e-10,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_gain_to_split': 0.0,
            'max_bin': 10000,
            'min_data_in_bin': 1,
            'bin_construct_sample_cnt': 1000000,
            'prep_params': {},
        },
        "linear_local": { # Local linearity
            "linear_tree": True,
            "num_leaves": 10,
            "min_data_in_leaf": 128, # TODO: Adjust adaptively
            "linear_lambda": 1,
            'prep_params': {},
        },

        "cat_all_onehot": { # cat treatment
            "max_cat_to_onehot": 100000,  # TODO: Make data-adaptive
            'prep_params': {},
        },
        'cat_fe': { # Preset for categorical feature engineering
            # TODO: Adjust to also operate on lower-cardinality features
            'prep_params': {
                'FrequencyEncoder': {},
                'CatGroupByAdder': {'min_cardinality': 6},
                'CatIntAdder': {'max_order': 3, 'add_freq': True, 'use_filters': False}, # TODO: Make sure not to forget target_type
            },
        },

        "extra_trees": { # extra_tree as the focus, need to think when and why this works well.
            "extra_trees": True,
            'prep_params': {},
        },

        "stumps": { # Datasets with no feature interactions
            "num_leaves": 2,
            'n_estimators': 100000, # TODO: Make sure that setting this actually works
            'prep_params': {},
        },

        "regularize_depth": { # XGB-like trees
            "max_depth": 4,
            'prep_params': {},

        },

        'RBFPCA': { 
            'prep_params': {'KernelPCAPreprocessor': {'kernel': 'rbf', 'only_numerical': False}},
        },            
        'FactorAnalysis': {
            'prep_params': {'FactorAnalysisPreprocessor': {'only_numerical': False}},
        },
        # Next promising candidates: 'FastICA', 'SigmoidPCA', 'SVD', SparsePCA

        ### Several arithmetic interaction presets
        'arithmetic_interactions_maxorder3_5000feats': {
            'prep_params': {'NumericalInteractionPreprocessor': {
                # 'target_type': target_type,
                'max_order': 3,
                'num_operations': 'all',
                'use_mvp': False,
                'corr_thresh': .95,
                'select_n_candidates': 5000,
                'apply_filters': False,
                # candidate_cols=candidate_cols,
                'min_cardinality': 3,
                }
            }
        },
        'arithmetic_by_spearman_depth2_20base': { # Selected arithmetic
            'prep_params': {'ArithmeticBySpearmanPreprocessor': {
                'operations': ["+", "-", "*", "/"],
                'correlation_threshold': 0.05,
                'max_depth': 2,
                'top_k': None,
                'base_k': 20,
                # 'target_type': target_type,
                }
            }
        },
        'arithmetic_scaled_interactions_maxorder3_2000feats': {
            'prep_params': {'NumericalInteractionPreprocessor': {
                # 'target_type': target_type,
                'scale_X': True,
                'max_order': 3,
                'num_operations': 'all',
                'use_mvp': False,
                'corr_thresh': .95,
                'select_n_candidates': 2000,
                'apply_filters': False,
                # candidate_cols=candidate_cols,
                'min_cardinality': 3,
                }
            }
        },
        'cat_as_num_arithmetic_2000feats_order3': { # cat_as_num should be a parameter when fine-tuning arithmetic interactions
            'prep_params': {
                'CatAsNumTransformer': {},
                'NumericalInteractionPreprocessor': {
                    # 'target_type': target_type,
                    'max_order': 3,
                    'num_operations': 'all',
                    'use_mvp': False,
                    'corr_thresh': .95,
                    'select_n_candidates': 2000,
                    'apply_filters': False,
                    # candidate_cols=candidate_cols,
                    'min_cardinality': 3,
                }
            }
        },
        'extra_trees_arithmetic_interactions_maxorder3_2000feats': {
            "extra_trees": True,
            'prep_params': {
                'NumericalInteractionPreprocessor': {
                    # 'target_type': target_type,
                    'max_order': 3,
                    'num_operations': 'all',
                    'use_mvp': False,
                    'corr_thresh': .95,
                    'select_n_candidates': 2000,
                    'apply_filters': False,
                    # candidate_cols=candidate_cols,
                    'min_cardinality': 3,
                }
            }
        },
        'arithmetic_interactions_maxorder3_2000feats_finegranularpatterns': {
            "min_data_in_leaf": 2,  
            "min_sum_hessian_in_leaf": 1e-5,
            "min_gain_to_split": 0.0,
            "min_data_per_group": 2, 
            'prep_params': {
                'NumericalInteractionPreprocessor': {
                    # 'target_type': target_type,
                    'max_order': 3,
                    'num_operations': 'all',
                    'use_mvp': False,
                    'corr_thresh': .95,
                    'select_n_candidates': 2000,
                    'apply_filters': False,
                    # candidate_cols=candidate_cols,
                    'min_cardinality': 3,
                }
            }
        },
        'all_as_cat_append_nocatreg': { 
            'cat_l2': 0,
            'cat_smooth': 0,
            'min_data_per_group': 1,
            'prep_params': {'ToCategoricalTransformer': {'keep_original': True, 'min_cardinality': 6}},
        },
        'groupby_interactions': { 
            'prep_params': {
                'GroupbyInteractionPreprocessor': {
                    # 'target_type': target_type,
                    'min_cardinality': 2,
                    'use_mvp': False,
                    'mean_difference': True,
                    'num_as_cat': False,
                }
            },
        },

        # TODO: Define a more general groupby preset
        # 'catnum_and_cat_groupby_interactions': { 
        #     'init_params': {},
        #     'preprocessors': [
        #         GroupbyInteractionPreprocessor(                    
        #             target_type = target_type,
        #             min_cardinality = 2,
        #             use_mvp = False,
        #             mean_difference = True,
        #             num_as_cat = False,
        #             ),
        #         CatGroupByAdder(min_cardinality=6),
        #         ],
        #     'cv_params': {},
        # }, 
        # 'cat_groupby_min_cardinality_6': { 
        #     'init_params': {},
        #     'preprocessors': [CatGroupByAdder(min_cardinality=6)],
        #     'cv_params': {},
        # },
        # 'cat_groupby_min_cardinality_3': { 
        #     'init_params': {},
        #     'preprocessors': [CatGroupByAdder(min_cardinality=3)], # TODO: Add filters since that likely doesn't make much sense
        #     'cv_params': {},
        # },

    ### EXPERIMENTL
        'linear_feature': { 
            'prep_params': {'LinearFeatureAdder': {}},
        },
        'cat_loo_append': { # aka & interactions of num features
            'prep_params': {'CatLOOTransformer': {'keep_original': True}},
        },
        'duplicate_count': { 
            'prep_params': {'DuplicateCountAdder': {}},
        },
        'duplicate_sample_loo': { 
            'prep_params': {'DuplicateSampleLOOEncoder': {}},
        },
        # 'duplicate_sample_loo': { 
        #     'metric': {'logloss': {}},
        # },

    }

    return presets