import numpy as np
import pandas as pd
from itertools import combinations 
from tabprep.utils.modeling_utils import make_cv_function
from tabprep.proxy_models import TargetMeanRegressor, TargetMeanClassifier, UnivariateLinearRegressor, UnivariateLogisticClassifier
from itertools import combinations

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

'''
Next steps:
- Move combination test to BasePreprocessor
- combination test as a method to assess the performance of interaction features
- Add interaction test to BasePreprocessor
'''

from tabprep.detectors.base_preprocessor import BasePreprocessor
class LinearTrendDetector(BasePreprocessor):
    def __init__(self, 
                 target_type, 
                n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                min_cardinality=6,
                combination_criterion='win',
                combination_test_min_bins=2,
                combination_test_max_bins=2048, #TODO: Test with 90% of unique values as max
                assign_numeric_with_combination=False,
                binning_strategy='lgb', # ['lgb', 'KMeans', 'DT']

                 ):
        # TODO: Include possibility to select operators
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.min_cardinality = min_cardinality
        self.combination_criterion = combination_criterion
        self.combination_test_min_bins = combination_test_min_bins
        self.combination_test_max_bins = combination_test_max_bins
        self.assign_numeric_with_combination = assign_numeric_with_combination
        self.binning_strategy = binning_strategy

        self.linear_features = {}

        self.get_linear_model = lambda x: Pipeline([
            # PowerTransformer removes a few more features to be numeric than standaradscaler, mostly the very imbalanced ones
            # ("standardize", PowerTransformer(method='yeo-johnson', standardize=True, )),
            ("standardize", QuantileTransformer(n_quantiles=np.min([1000, int(x.shape[0]*(1-(1/self.n_folds)))]), random_state=42)),
            ("impute", SimpleImputer(strategy="median")),
            # ("standardize", StandardScaler()),
            ("model", UnivariateLinearRegressor() if self.target_type == 'regression' else UnivariateLogisticClassifier())
        ])

    
    def adapt_for_mvp_test(self, X_cand_in, test_cols, col=None, mode='forward'):
        if col is None:
            X_out = X_cand_in.copy()
            # NOTE: 'backward' in full scenario should always correspond to using the raw data version.
                # If features have been added they should be dropped, if nothing was changed, the raw data should be used.
            if mode == 'backward': 
                pass
            elif mode == 'forward':
                for col in test_cols:
                        if self.target_type=='regression':
                            X_out[col+'_linearized'] = self.linear_features[col].predict(X_out[col].to_frame())
                        else:
                            X_out[col+'_linearized'] = self.linear_features[col].predict_proba(X_out[col].to_frame())[:,1]
        else:
            X_out = X_cand_in.copy()
            if mode == 'backward':
                for use_col in test_cols:
                    if col != use_col:
                        if self.target_type=='regression':
                            X_out[use_col+'_linearized'] = self.linear_features[use_col].predict(X_out[use_col].to_frame())
                        else:
                            X_out[use_col+'_linearized'] = self.linear_features[use_col].predict_proba(X_out[use_col].to_frame())[:,1]
            elif mode == 'forward':
                for use_col in test_cols:
                    if col == use_col:
                        if self.target_type=='regression':
                            X_out[use_col+'_linearized'] = self.linear_features[use_col].predict(X_out[use_col].to_frame())
                        else:
                            X_out[use_col+'_linearized'] = self.linear_features[use_col].predict_proba(X_out[use_col].to_frame())[:,1]

        return X_out

    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        y = self.adjust_target_format(y)

        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(exclude=[np.number])

        candidate_cols = X_num.columns[(X_num.nunique() >= self.min_cardinality).values].tolist()
        if len(candidate_cols) == 0:
            return self

        self.get_dummy_mean_scores(X_num[candidate_cols], y)
        # TODO: Add filter to compare single results to LGB on the full dataset
        # TODO: Test logic to start with the column that is strongest for linear and try only for that one
        for cnum, col in enumerate(candidate_cols):         
            # TODO: Add detector for target encoding to be used whenever we have a very strong signal for that (single-feature-mean>lgb)   
            if self.verbose:
                print(f"\r{cnum}/{len(candidate_cols)} columns processed", end="", flush=True)
            x_use = X_num[col].copy()
            self.scores[col]['linear'] = self.single_interpolation_test(x_use, y, interpolation_method='linear')

            self.significances[col][f"significant_linear"] = self.significance_test(
                        self.scores[col]['linear'] - self.scores[col]["dummy"]
                    )

            if self.significances[col][f"significant_linear"] > self.alpha:
                continue

            self.significances[col][f"test_linear_superior_mean"] = self.significance_test(
                        self.scores[col]['linear'] - self.scores[col]["mean"]
                    ) 
            if self.significances[col][f"test_linear_superior_mean"] > self.alpha:
                continue
            # else:
            #     self.linear_features[col] = self.get_linear_model(x_use).fit(x_use.to_frame(), y)

            any_comb_superior = False
            m_bins = [2**i for i in range(1,100) if 2**i>= self.combination_test_min_bins and 2**i <= self.combination_test_max_bins]
            for m_bin in m_bins:
                ### Combination test
                nunique = x_use.nunique()
                if m_bin > nunique:
                    continue
                # if nunique>10000:
                #     m_bin = int(10000*q)
                # else:
                #     m_bin = int(nunique*q)

                self.scores[col][f"combination_test_{m_bin}"] = self.single_combination_test(x_use, y, max_bin=m_bin, binning_strategy=self.binning_strategy)
                self.significances[col][f"test_linear_superior_combination_{m_bin}"] = self.significance_test(
                        self.scores[col]['linear'] - self.scores[col][f"combination_test_{m_bin}"]
                    ) 
                if self.significances[col][f"test_linear_superior_combination_{m_bin}"] > self.alpha:
                    any_comb_superior = True
                    break
            if not any_comb_superior:
                self.linear_features[col] = self.get_linear_model(x_use).fit(x_use.to_frame(), y)


        if len(self.linear_features)>0:
            X_int = self.multivariate_performance_test(X, y,
                                                    test_cols=list(self.linear_features.keys()), 
                                                    suffix='LINEAR',
                                                    )

        return self
    
    # def transform(self, X):
    #     X_out = X.copy()

    #     return X_out
