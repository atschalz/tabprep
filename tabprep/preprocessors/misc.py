from tabprep.preprocessors.base import BasePreprocessor
from tabprep.proxy_models import (
    TargetMeanRegressor, 
    TargetMeanClassifier,
    UnivariateLinearRegressor,
    UnivariateLogisticClassifier,
    TargetMeanRegressorCut,
    TargetMeanClassifierCut
)

class TargetRepresenter(BasePreprocessor):
    def __init__(self, feature_target_rep, target_type):
        super().__init__(keep_original=False)
        self.feature_target_rep = feature_target_rep.copy()
        self.target_type = target_type

        self.col_transformer = {}

        if self.target_type == 'binary':
            self.mean_transformer = TargetMeanClassifier
            self.linear_transformer = UnivariateLogisticClassifier
            self.mean_threshold_transformer = TargetMeanClassifierCut
        elif self.target_type == 'regression':
            self.mean_transformer = TargetMeanRegressor
            self.linear_transformer = UnivariateLinearRegressor
            self.mean_threshold_transformer = TargetMeanRegressorCut

        self.binner = LightGBMBinner
    
    def correlation_filter(self, X_in, threshold=0.95):
        corr_matrix = X_in.corr(method='spearman').abs()
        # Drop highly correlated features
        reduced_corr_matrix = drop_highly_correlated_features(corr_matrix, threshold)
        return X_in[reduced_corr_matrix.columns]
    
    def fit(self, X_in, y_in):
        X = X_in.copy()
        y = y_in.copy()
        self.drop_cols = []
        for col in X.columns:
            if col not in self.feature_target_rep:
                # FIXME: Currently, we assign best treatment outside of the CV. This sometimes leads to different features being filtered inside the CV. A solution could be less strong filters.
                continue
            if self.feature_target_rep[col] == 'raw':
                continue
            elif self.feature_target_rep[col] == 'mean':
                self.col_transformer[col] = Pipeline([
                    ('dtype', CategoricalDtypeAssigner()),
                    ('loo', LeaveOneOutEncoder()),
                    ('model', self.mean_transformer())
                ])
                self.col_transformer[col].fit(X[[col]], y)
            elif self.feature_target_rep[col] == 'linear':
                self.col_transformer[col] = self.linear_transformer()
                self.col_transformer[col].fit(X[[col]], y)
            elif 'combination_test' in self.feature_target_rep[col]:
                bins = int(self.feature_target_rep[col].split('_')[-1])
                self.col_transformer[col] = self.binner(bins)
                self.col_transformer[col] = Pipeline([
                    ('binner', self.binner(bins)),
                    ('model', self.mean_transformer()) # TODO: Test with LOO
                    ])
                self.col_transformer[col].fit(X[[col]], y)
            elif 'cat_threshold' in self.feature_target_rep[col]:
                bins = int(self.feature_target_rep[col].split('_')[-1])
                self.col_transformer[col] = self.mean_threshold_transformer(bins) # TODO: Test with LOO
                self.col_transformer[col].fit(X[[col]], y)
            elif self.feature_target_rep[col] == 'dummy':
                self.drop_cols.append(col)
            else:
                raise ValueError(f"Unknown feature_target_rep value: {self.feature_target_rep[col]} for column {col}")
            
            
            # if self.feature_target_rep[col] == 'mean':
            #     X[col] = self.col_transformer[col].transform(X[[col]])
            # else:
            X[col] = self.col_transformer[col].predict(X[[col]]) if self.target_type == 'regression' else self.col_transformer[col].predict_proba(X[[col]])[:,1]

        X_new = self.correlation_filter(X, threshold=0.95)  # Apply correlation filter
        
        self.drop_cols.extend(list(set(X.columns) - set(X_new.columns)))

        return self
    
    def transform(self, X_in):
        X = X_in.copy()
        X = X.drop(self.drop_cols, axis=1)
        for col in X.columns:
            if col in self.col_transformer:
                # if self.feature_target_rep[col] == 'mean':
                #     X[col] = self.col_transformer[col].transform(X[[col]])
                # else:
                X[col] = self.col_transformer[col].predict(X[[col]]) if self.target_type == 'regression' else self.col_transformer[col].predict_proba(X[[col]])[:,1]
        
        X = X.drop(columns=self.drop_cols, errors='ignore')
        return X