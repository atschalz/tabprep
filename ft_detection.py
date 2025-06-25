import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error, log_loss
from utils import TargetMeanClassifier,TargetMeanRegressor,UnivariateLinearRegressor,UnivariateLogisticClassifier,PolynomialLogisticClassifier,PolynomialRegressor, p_value_wilcoxon_greater_than_zero, clean_series, UnivariateThresholdClassifier, MultiFeatureTargetMeanClassifier, MultiFeatureUnivariateLogisticClassifier, make_cv_scores_with_early_stopping
from sklearn.linear_model import LogisticRegression
import os
import time

class FeatureTypeDetector(TransformerMixin, BaseEstimator):
    def __init__(self, target_type, min_q_as_num=6, n_folds=5, lgb_model_type="unique-based", assign_numeric=False, detect_numeric_in_string=False, use_highest_corr_feature=False, num_corr_feats_use=0):

        self.target_type = target_type
        self.min_q_as_num = min_q_as_num
        self.n_folds = n_folds
        self.lgb_model_type = lgb_model_type
        self.assign_numeric = assign_numeric
        self.detect_numeric_in_string = detect_numeric_in_string
        self.use_highest_corr_feature = use_highest_corr_feature
        self.num_corr_feats_use = num_corr_feats_use
        self.reassigned_features = []
        self.cat_dtype_maps = {}
        # TODO: Fix multi-class behavior
        # TODO: Think again whether the current proxy model really is the best choice (Might use a small NN instead of LGBM; Might use different HPs )
        # TODO: Implement the parallelization speedup for all interpolation tests 
        # TODO: Track how many features have been filtered by which step (mainly needs adding this for interpolation and combination tests)

    def fit(self, X_input, y_input=None, verbose=False):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)
        if not isinstance(y_input, pd.DataFrame):
            y_input = pd.Series(y_input)
        
        self.orig_dtypes = {col: "categorical" if dtype in ["object", "category", str] else "numeric" for col,dtype in dict(X_input.dtypes).items()}

        X = X_input.copy()
        y = y_input.copy()

        if self.target_type=="multiclass":
            # TODO: Fix this hack
            y = (y==y.value_counts().index[0]).astype(int)  # make it binary
            self.target_type = "binary"

        self.col_names = X.columns
        self.dtypes = {col: "None" for col in  self.col_names}
        self.significances = {}
        self.scores = {}

        ### 1. Assign binary type (assume uniform columns have not been handled before and assign them to be binary)
        ### 2. Handle low-cardinality - (OHE? Leave as cat?)
        bin_cols = []
        lowcard_cols = []
        for col in self.col_names:
            if X[col].nunique()<=2:
                self.dtypes[col] = "binary"
                bin_cols.append(col)
            elif X[col].nunique()<self.min_q_as_num:
                self.dtypes[col] = "low-cardinality"
                lowcard_cols.append(col)
        
        if verbose:
            print(f"{len(bin_cols)}/{len(self.col_names)} columns are binary")        
        rem_cols = [x for x in self.col_names if x not in bin_cols]
        
        if len(bin_cols) == len(self.col_names):
            return 
        if verbose:
            print(f"{len(lowcard_cols)}/{len(rem_cols)} columns are low-cardinality")        
        rem_cols = [x for x in rem_cols if x not in lowcard_cols]
        
        ### 3. Clearly categoricals
        num_coerced = np.array([int(pd.to_numeric(X[col].dropna(), errors= 'coerce').isna().sum()) for col in rem_cols])
        
        numeric_cand_cols = np.array(rem_cols)[num_coerced==0].tolist()
        if verbose:
            print(f"{len(numeric_cand_cols)}/{len(num_coerced)} columns can be converted to floats")        
        rem_cols = [x for x in rem_cols if x not in numeric_cand_cols]
        
        num_coerced = np.array([int(pd.to_numeric(X[col].dropna(), errors= 'coerce').isna().sum()) for col in rem_cols])
        if self.detect_numeric_in_string: # Currently doesn't work!!!! - see TODO below
            cat_cols = np.array(rem_cols)[[num_coerced[num]==X[col].dropna().shape[0] for num, col in enumerate(rem_cols)]].tolist()
        else:
            cat_cols = np.array(rem_cols)[[num_coerced[num]<=X[col].dropna().shape[0] for num, col in enumerate(rem_cols)]].tolist()
        
        if verbose:
            print(f"{len(cat_cols)}/{len(num_coerced)} columns are entirely categorical")        
        rem_cols = [x for x in rem_cols if x not in cat_cols]
        for col in cat_cols:
            self.dtypes[col] = "categorical"
        
        # 4. Try to extract numerical features from string columns
        # TODO: To apply this, we would need to integrate it in the fit-transform logic. Therefore, disable for now.
        # if len(rem_cols)>0:
        #     num_coerced = np.array([int(pd.to_numeric(X[col].dropna(), errors= 'coerce').isna().sum()) for col in rem_cols])
        #     part_coerced = np.array(rem_cols)[[0<c<X[col].dropna().shape[0] for c, col in zip(num_coerced, rem_cols)]].tolist()
        #     if len(part_coerced)>0:
        #         X_copy = X.loc[:, part_coerced].apply(clean_series)
        #         X[part_coerced] = X_copy
        #     all_nan = X_rem.columns[X_rem.isna().mean()==1]
        #     if verbose:
        #         print(f"{len(part_coerced)}/{len(rem_cols)} columns are partially numerical. {len(all_nan)} of them don't show regular patterns and are treated as categorical.")        
            
        #     if len(all_nan)>0:
        #         for col in all_nan:
        #             self.dtypes[col] = "categorical"
        #     rem_cols = [x for x in rem_cols if x not in part_coerced]
        #     numeric_cand_cols += [x for x in part_coerced if x not in all_nan]
            
        assert len(rem_cols)==0


        ### 5. Interpolation test
        if len(numeric_cand_cols)>0:
            print("------------")
            X_rem = X[numeric_cand_cols].copy().astype(float) # TODO: Assure that .asfloat() is sufficient to correctly treat the columns as numeric

            print(f"{len(numeric_cand_cols)} columns left for numeric/categorical detection")

            if self.target_type == "binary":
                scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
                model_class = lgb.LGBMClassifier
                dummy = DummyClassifier(strategy='prior')
                linear =  UnivariateLogisticClassifier() #PolynomialLogisticClassifier(degree=1) #UnivariateLogisticClassifier() # LogisticRegression(solver='liblinear')
                poly_2 = PolynomialLogisticClassifier(degree=2)
                poly_3 = PolynomialLogisticClassifier(degree=3)
                poly_4 = PolynomialLogisticClassifier(degree=4)
                poly_5 = PolynomialLogisticClassifier(degree=5)
                cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            else:
                scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
                model_class = lgb.LGBMRegressor
                dummy = DummyRegressor(strategy='mean')
                linear = UnivariateLinearRegressor()
                poly_2 = PolynomialRegressor(degree=2)
                poly_3 = PolynomialRegressor(degree=3)
                poly_4 = PolynomialRegressor(degree=4)
                poly_5 = PolynomialRegressor(degree=5)
                cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            cv_scores_with_early_stopping = make_cv_scores_with_early_stopping(target_type=self.target_type, scorer=scorer, cv=cv, early_stopping_rounds=20, vectorized=False)
            cv_scores_with_early_stopping_vec = make_cv_scores_with_early_stopping(target_type=self.target_type, scorer=scorer, cv=cv, early_stopping_rounds=20, vectorized=True)
            
            base_params = {
                "objective": "binary" if self.target_type=="binary" else "regression",
                "boosting_type": "gbdt",
                # "n_estimators": 1000,
                "verbosity": -1
            }

            # res_df = pd.DataFrame() #index=["dummy"] + modes)
            # assignments = {}

            # Fit model on full data - currently optional, likely not needed.
            # params = base_params.copy()
            # params["max_bin_by_feature"] = (X_rem.nunique()+1).tolist()
            # full_pipe = Pipeline([("model", model_class(**params))])
            # full_scores = cv_scores_with_early_stopping(X_rem, y, full_pipe)
            # res_df["full"] = round(full_scores.mean(), 6)

            # Linear Interpolation test
            linear_interpol_cols = []
            # TODO: Parallelize Target mean detection for regression
            if self.target_type == "regression":
                for cnum, col in enumerate(X_rem.columns):
                    self.scores[col] = {}
                    print(f"\rLinear interpolation test: {cnum+1}/{len(X_rem.columns)} columns processed", end="", flush=True)
                    self.significances[col] = {}

                    # dummy baseline on single column (currently not used)
                    dummy_pipe = Pipeline([("model", dummy)])
                    self.scores[col]["dummy"] = cv_scores_with_early_stopping(X_rem[[col]], y, dummy_pipe)

                    x_use = X_rem[col].copy()
                    
                    model = (TargetMeanClassifier() if self.target_type=="binary"
                            else TargetMeanRegressor())
                    pipe = Pipeline([("model", model)])
                    self.scores[col]["mean"] = cv_scores_with_early_stopping(x_use.to_frame(), y, pipe)

                    pipe = Pipeline([
                        ("impute", SimpleImputer(strategy="median")), 
                        ("standardize", StandardScaler(
                        )),
                        ("model", linear)
                    ])
                    # if cnum==0:
                    #     print(pipe)
                    self.scores[col]["linear"] = cv_scores_with_early_stopping(x_use.to_frame(), y, pipe)
    
                    # self.significances[col]["test_irrelevant_mean"] = p_value_wilcoxon_greater_than_zero(
                    #     self.scores["dummy"] - self.scores["mean"]
                    # )
                    # self.significances[col]["test_irrelevant_linear"] = p_value_wilcoxon_greater_than_zero(
                    #             self.scores["dummy"] - self.scores["linear"]
                    #         )
                    self.significances[col]["test_linear_superior"] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col]["linear"] - self.scores[col]["mean"]
                    )
                    
                    if self.significances[col]["test_linear_superior"]<0.05:
                        self.dtypes[col] = "numeric"

                        linear_interpol_cols.append(col)
            else:
                pipe = Pipeline([("model", MultiFeatureTargetMeanClassifier())])
                mean_res = cv_scores_with_early_stopping_vec(X_rem, y, pipe)
                
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")), 
                    ("standardize", StandardScaler(
                    )),
                    ("model", MultiFeatureUnivariateLogisticClassifier())
                ])
                # if cnum==0:
                #     print(pipe)
                linear_res = cv_scores_with_early_stopping_vec(X_rem, y, pipe)

                for cnum, col in enumerate(numeric_cand_cols):
                    x_use = X_rem[col].copy()
                    self.scores[col] = {}
                    print(f"\rLinear interpolation test: {cnum}/{len(numeric_cand_cols)} columns processed", end="", flush=True)
                    self.significances[col] = {}

                    # dummy baseline on single column 
                    dummy_pipe = Pipeline([("model", dummy)])
                    self.scores[col]["dummy"] = cv_scores_with_early_stopping(x_use.to_frame(), y, dummy_pipe)
                    
                    # model = (TargetMeanClassifier() if self.target_type=="binary"
                    #         else TargetMeanRegressor())
                    # pipe = Pipeline([("model", model)])
                    # self.scores["mean"] = cv_scores_with_early_stopping(x_use.to_frame(), y, pipe)
                
                    self.scores[col]["mean"] = np.array([mean_res[fold][col] for fold in range(self.n_folds)])
                    self.scores[col]["linear"] = np.array([linear_res[fold][col] for fold in range(self.n_folds)])
                    # self.significances[col]["test_irrelevant_mean"] = p_value_wilcoxon_greater_than_zero(
                    #     self.scores["dummy"] - self.scores["mean"]
                    # )
                    # self.significances[col]["test_irrelevant_linear"] = p_value_wilcoxon_greater_than_zero(
                    #             self.scores["dummy"] - self.scores["linear"]
                    #         )
                    self.significances[col]["test_linear_superior"] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col]["linear"] - self.scores[col]["mean"]
                    )
                    
                    if self.significances[col]["test_linear_superior"]<0.05:
                        self.dtypes[col] = "numeric"

                        linear_interpol_cols.append(col)
            
            if verbose:
                print("\n")
                print(f"{len(linear_interpol_cols)}/{len(numeric_cand_cols)} columns are numeric acc. to linear interpolation test.")        
            
            numeric_cand_cols = [x for x in numeric_cand_cols if x not in linear_interpol_cols]
            
            if len(numeric_cand_cols)==0:
                return 

            # TODO: Rename X_rem to X_num (but verify that this is what it represents)
            # Linear interpolation test for highest corr feature
            if self.use_highest_corr_feature:
                
                n_cols_start = len(numeric_cand_cols)
                linear_interpol_cols_feat = []
                for n_col in range(min([self.num_corr_feats_use, X_rem.shape[1]-1])):
                    linear_feat = UnivariateLinearRegressor()
                    scorer_feat = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr)
                    cv_feat = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
                    cv_scores_with_early_stopping_feat = make_cv_scores_with_early_stopping(target_type="regression", scorer=scorer_feat, cv=cv_feat, early_stopping_rounds=20, vectorized=False)

                    for cnum, col in enumerate(numeric_cand_cols):
                    
                        print(f"\rLinear interpolation test ({n_col+1}-high-corr feature): {cnum+1}/{len(numeric_cand_cols)} columns processed", end="/n", flush=False)
                        x_use = X_rem[col].copy()
                        x_use = x_use.fillna(x_use.mean()).astype(float)

                        # TODO: On some datasets, sometimes an annoying warning appears - make sure that doesnt happen
                        filtered_df = X_rem.drop(col, axis=1)
                        y_use = filtered_df[filtered_df.corrwith(x_use, drop=True).abs().sort_values(ascending=False).index[n_col]]
                        y_use = y_use.fillna(y_use.mean()).astype(float)

                        y_use = pd.Series(MinMaxScaler().fit_transform(y_use.values.reshape(-1, 1)).flatten(), index=y_use.index, name= y_use.name)

                        model = TargetMeanRegressor()
                        pipe = Pipeline([("model", model)])
                        self.scores[col][f"mean-feat-{n_col+1}"] = cv_scores_with_early_stopping_feat(x_use.to_frame(), y_use, pipe)

                        pipe = Pipeline([
                            ("impute", SimpleImputer(strategy="median")), 
                            ("standardize", StandardScaler(
                            )),
                            ("model", linear_feat)
                        ])
                        self.scores[col][f"linear-feat-{n_col+1}"] = cv_scores_with_early_stopping_feat(x_use.to_frame(), y_use, pipe)

                        self.significances[col][f"test_linear-feat_superior-{n_col+1}"] = p_value_wilcoxon_greater_than_zero(
                            self.scores[col][f"linear-feat-{n_col+1}"] - self.scores[col][f"mean-feat-{n_col+1}"]
                        )

                        if self.significances[col][f"test_linear-feat_superior-{n_col+1}"]<0.05:
                            self.dtypes[col] = "numeric"
                            linear_interpol_cols_feat.append(col)
                    numeric_cand_cols = [x for x in numeric_cand_cols if x not in linear_interpol_cols_feat]
                    if len(numeric_cand_cols)==0:
                        break
                if verbose:
                    print("\n")
                    print(f"{len(linear_interpol_cols_feat)}/{n_cols_start} columns are numeric acc. to linear interpolation test on highest correlated feature.")

                if len(numeric_cand_cols)==0:
                    return 

            # Interpolation test for polynomial of degree=2
            poly2_interpol_cols = []

            for cnum, col in enumerate(numeric_cand_cols):                
                print(f"\r2-polynomial interpolation test: {cnum+1}/{len(numeric_cand_cols)} columns processed", end="", flush=True)
                x_use = X_rem[col].copy()
                
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")), 
                    ("standardize", StandardScaler(
                    )),
                    ("model", poly_2)
                ])
                self.scores[col]["poly2"] = cv_scores_with_early_stopping(x_use.to_frame(), y, pipe)
   
                self.significances[col]["test_poly2_superior"] = p_value_wilcoxon_greater_than_zero(
                    self.scores[col]["poly2"] - self.scores[col]["mean"]
                )
                
                if self.significances[col]["test_poly2_superior"]<0.05:
                    self.dtypes[col] = "numeric"
                    poly2_interpol_cols.append(col)
                                    

            if verbose:
                print("\n")
                print(f"{len(poly2_interpol_cols)}/{len(numeric_cand_cols)} columns are numeric acc. to 2-polynomial interpolation test.")        
            

            numeric_cand_cols = [x for x in numeric_cand_cols if x not in poly2_interpol_cols]
            
            if len(numeric_cand_cols)==0:
                return 

            # Interpolation test for polynomial of degree=3
            poly3_interpol_cols = []

            for cnum, col in enumerate(numeric_cand_cols):                
                print(f"\r3-polynomial interpolation test: {cnum+1}/{len(numeric_cand_cols)} columns processed", end="", flush=True)
                x_use = X_rem[col].copy()
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")), 
                    ("standardize", StandardScaler(
                    )),
                    ("model", poly_3)
                ])
                self.scores[col]["poly3"] = cv_scores_with_early_stopping(x_use.to_frame(), y, pipe)

                # TODO: Verify that the testing function works as intended
                # TODO: Add hyperparameter for alternatives to wilcoxon rank test
                self.significances[col]["test_poly3_superior"] = p_value_wilcoxon_greater_than_zero(
                    self.scores[col]["poly3"] - self.scores[col]["mean"]
                )
                
                if self.significances[col]["test_poly3_superior"]<0.05:
                    self.dtypes[col] = "numeric"
                    poly3_interpol_cols.append(col)
                                    

            if verbose:
                print("\n")
                print(f"{len(poly3_interpol_cols)}/{len(numeric_cand_cols)} columns are numeric acc. to 3-polynomial interpolation test.")        

            numeric_cand_cols = [x for x in numeric_cand_cols if x not in poly3_interpol_cols]
            
            if len(numeric_cand_cols)==0:
                return  

            # TODO: Test whether polynomials of a higher degree can be useful

            # TODO: Combination test likely is only required if there are still numeric candidate columns left that were not already classified as categorical by the previous tests
            ### Combination test
            # TODO: Test whether using a sklearn decision tree can speed things up
            comb_num = []
            comb_cat = []
            comb_bothfine = []
            irrelevant_cols = []
            for cnum, col in enumerate(numeric_cand_cols):                
                print(f"\rCombination test: {cnum+1}/{len(numeric_cand_cols)} columns processed", end="", flush=True)
                x_use = X_rem[[col]].copy()
                
                # TODO: Make a function as it reappears
                if self.lgb_model_type=="unique-based":
                    params = base_params.copy()
                    params["max_bin"] = X_rem[col].nunique() #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                    params["max_depth"] = 2 #if mode=="cat" else 2
                    params["n_estimators"] = X_rem[col].nunique() # 10000 #min(max(int(X[col].nunique()/4),1),100)
                elif self.lgb_model_type=="unique-based-binned":
                    params = base_params.copy()
                    params["max_bin"] = int(X_rem[col].nunique()/2) #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                    params["max_depth"] = 2 #if mode=="cat" else 2
                    params["n_estimators"] = X_rem[col].nunique() # 10000 #min(max(int(X[col].nunique()/4),1),100)
                elif self.lgb_model_type=="high-capacity":
                    params = base_params.copy()
                    params["max_bin"] = X_rem[col].nunique() #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                    params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)
                elif self.lgb_model_type=="high-capacity-binned":
                    params = base_params.copy()
                    params["max_bin"] = int(X_rem[col].nunique()/2) #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                    params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)
                elif self.lgb_model_type=="default":
                    params = base_params.copy()
                else:
                    raise ValueError(f"Unknown lgb_model_type: {self.lgb_model_type}. Use 'unique-based', 'high-capacity', 'high-capacity-binned' or 'default'.")

                    
                model = model_class(**params)
                pipe = Pipeline([("model", model)])
                self.scores[col]["lgb"] = cv_scores_with_early_stopping(x_use, y, pipe)


                self.significances[col]["test_lgb_superior"] = p_value_wilcoxon_greater_than_zero(
                    self.scores[col]["lgb"] - self.scores[col]["mean"]
                )
                
                self.significances[col]["test_mean_superior"] = p_value_wilcoxon_greater_than_zero(
                    self.scores[col]["mean"] - self.scores[col]["lgb"]
                )
                
                self.significances[col]["test_lgb_beats_dummy"] = p_value_wilcoxon_greater_than_zero(
                    self.scores[col]["lgb"] - self.scores[col]["dummy"]
                )
                
                self.significances[col]["test_mean_beats_dummy"] = p_value_wilcoxon_greater_than_zero(
                    self.scores[col]["lgb"] - self.scores[col]["dummy"]
                )

                # Conservative version
                # TODO: Experiment with different ways to determine categorical features
                ### !! We only want to infer anything about the column if it is possible to predict better than the dummy classifier
                if self.significances[col]["test_lgb_beats_dummy"]<0.05 and self.significances[col]["test_mean_beats_dummy"]<0.05:
                    if self.significances[col]["test_mean_superior"]<0.05:
                        self.dtypes[col] = "categorical"
                        comb_cat.append(col)
                    elif self.significances[col]["test_lgb_superior"]<0.05:
                        self.dtypes[col] = "numeric"
                        comb_num.append(col)
                    else:
                        self.dtypes[col] = "indifferent" 
                        comb_bothfine.append(col)
                else:
                    self.dtypes[col] = "irrelevant"
                    irrelevant_cols.append(col)
       
            if verbose:
                print("\n")
                print(f"According to the combination test:")
                print(f"...   {len(irrelevant_cols)}/{len(numeric_cand_cols)} columns are not predictive at all.")    
                print(f"...   {len(comb_num)}/{len(numeric_cand_cols)} columns are numeric.")    
                print(f"...   {len(comb_cat)}/{len(numeric_cand_cols)} columns are categorical.")    
                print(f"...   {len(comb_bothfine)}/{len(numeric_cand_cols)} columns are indifferent.")    

            ### Reality check: Can transforming to categorical improve performance?
            cat_improve_cols = []
            for cnum, col in enumerate(comb_cat):                
                print(f"\rAs-categorical test: {cnum+1}/{len(comb_cat)} columns processed", end="", flush=True)
                x_use = X_rem[[col]].copy()
                if self.lgb_model_type=="unique-based":
                    params = base_params.copy()
                    params["max_bin"] = X_rem[col].nunique() #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                    params["max_depth"] = 2 #if mode=="cat" else 2
                    params["n_estimators"] = X_rem[col].nunique() # 10000 #min(max(int(X[col].nunique()/4),1),100)
                elif self.lgb_model_type=="unique-based-binned":
                    params = base_params.copy()
                    params["max_bin"] = int(X_rem[col].nunique()/2) #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                    params["max_depth"] = 2 #if mode=="cat" else 2
                    params["n_estimators"] = X_rem[col].nunique() # 10000 #min(max(int(X[col].nunique()/4),1),100)
                elif self.lgb_model_type=="high-capacity":
                    params = base_params.copy()
                    params["max_bin"] = X_rem[col].nunique() #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                    params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)
                elif self.lgb_model_type=="high-capacity-binned":
                    params = base_params.copy()
                    params["max_bin"] = int(X_rem[col].nunique()/2) #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                    params["n_estimators"] = 10000 #min(max(int(X[col].nunique()/4),1),100)
                elif self.lgb_model_type=="default":
                    params = base_params.copy()
                else:
                    raise ValueError(f"Unknown lgb_model_type: {self.lgb_model_type}. Use 'unique-based', 'high-capacity', 'high-capacity-binned' or 'default'.")


                model = model_class(**params)
                pipe = Pipeline([("model", model)])
                self.scores[col]["lgb-cat"] = cv_scores_with_early_stopping(x_use.astype("category"), y, pipe)


                self.significances[col]["test_cat_superior"] = p_value_wilcoxon_greater_than_zero(
                    self.scores[col]["lgb-cat"] - self.scores[col]["lgb"]
                )

                if self.significances[col]["test_cat_superior"]<0.05:
                    cat_improve_cols.append(col)
                else:
                    self.dtypes[col] = "mean>lgb>=lgb-cat" 

            if verbose:
                print("\n")
                print(f"For {len(cat_improve_cols)}/{len(comb_cat)} columns performance can be improved by transforming to categorical.")
            
            # Prepare objects to transform columns
            reassign_cols = [col for col in X.columns if self.dtypes[col]=="categorical" and self.orig_dtypes[col]!="categorical"]
            for col in reassign_cols:
                self.cat_dtype_maps[col] = pd.CategoricalDtype(categories=list(X[col].astype(str).fillna("nan").unique()))
                # TODO: CHange to use only train
            
            if self.assign_numeric:
                reassign_cols = [col for col in X.columns if self.dtypes[col]=="numeric" and self.orig_dtypes[col]!="numeric"]
                self.numeric_means = {col: X_rem[col].mean() for col in reassign_cols}

            return 
        else:
            return 

    def transform(self, X_input):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input)

        X = X_input.copy()

        for col in self.cat_dtype_maps:
            X[col] = X[col].astype(str).fillna("nan").astype(self.cat_dtype_maps[col])

        if self.assign_numeric:
            reassign_cols = [col for col in X.columns if self.dtypes[col]=="numeric" and self.orig_dtypes[col]!="numeric"]
            for col in reassign_cols:
                # TODO: Implement functionality for partially coerced columns
                X[col] = pd.to_numeric(X[col], errors= 'coerce').astype(float)
                if X[col].isna().any() and self.orig_dtypes==[col]=="categorical":
                    X[col] = X[col].fillna(self.numeric_means[col])

        return X
