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
    def __init__(self, target_type, min_q_as_num=6, n_folds=5):
        # TODO: Add hyperparameter for making numeric in string detection optional
        # TODO: Add a hyperparameter to optionally also transform initially categorical features to numeric
        # TODO: Fix multi-class behavior
        # TODO: Think again whether the current proxy model really is the best choice (Might use a small NN instead of LGBM; Might use different HPs )
        # TODO: Implement the parallelization speedup for all interpolation tests 
        self.target_type = target_type
        self.min_q_as_num=min_q_as_num
        self.n_folds=n_folds
        self.reassigned_features = []
        self.cat_dtype_maps = {}

    def fit(self, X, y=None, verbose=False):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        #TODO: Need copy here???

        self.orig_dtypes = {col: "numeric" if dtype in [int, float] else "categorical" for col,dtype in dict(X.dtypes).items()}

        self.col_names = X.columns
        self.dtypes = {col: None for col in  self.col_names}

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
        X_rem = X[rem_cols].copy()
        num_coerced = np.array([int(pd.to_numeric(X_rem[col].dropna(), errors= 'coerce').isna().sum()) for col in X_rem.columns])
        
        numeric_cand_cols = X_rem.columns[num_coerced==0].values.tolist()
        cat_cols = X_rem.columns[num_coerced==X_rem.shape[0]].values.tolist()

        if verbose:
            print(f"{len(numeric_cand_cols)}/{len(rem_cols)} columns can be converted to floats")        
        rem_cols = [x for x in rem_cols if x not in numeric_cand_cols]

        if verbose:
            print(f"{len(cat_cols)}/{len(rem_cols)} columns are entirely categorical")        
        rem_cols = [x for x in rem_cols if x not in cat_cols]
        for col in cat_cols:
            self.dtypes[col] = "categorical"

        if len(rem_cols)>0:
            part_coerced = X_rem.columns[[0<c<X_rem.shape[0] for c in num_coerced]].values.tolist()
            if len(part_coerced)>0:
                X_copy = X_rem.loc[:, part_coerced].apply(clean_series)
                X_rem = X_rem.drop(columns=part_coerced)
                X_rem[part_coerced] = X_copy
            all_nan = X_rem.columns[X_rem.isna().mean()==1]
            if verbose:
                print(f"{len(part_coerced)}/{len(rem_cols)} columns are partially numerical. {len(all_nan)} of them don't show regular patterns and are treated as categorical.")        
            
            if len(all_nan)>0:
                for col in all_nan:
                    self.dtypes[col] = "categorical"
            rem_cols = [x for x in rem_cols if x not in part_coerced]
            numeric_cand_cols += [x for x in part_coerced if x not in all_nan]
            
        assert len(rem_cols)==0


        ### 4. Interpolation test
        if len(numeric_cand_cols)>0:
            self.significances = {}
            self.scores = {}
            print("------------")
            X_rem = X_rem[numeric_cand_cols].astype(float)



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
                "n_estimators": 1000,
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
            # TODO: Parallelize Trget mean detection for regression
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

                for cnum, col in enumerate(X_rem.columns):
                    self.scores[col] = {}
                    print(f"\rLinear interpolation test: {cnum}/{len(X_rem.columns)} columns processed", end="", flush=True)
                    self.significances[col] = {}

                    # dummy baseline on single column (currently not used)
                    dummy_pipe = Pipeline([("model", dummy)])
                    self.scores[col]["dummy"] = cv_scores_with_early_stopping(X_rem[[col]], y, dummy_pipe)

                    x_use = X_rem[col].copy()
                    
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
            
            X_rem = X_rem.drop(linear_interpol_cols,axis=1)
            numeric_cand_cols = [x for x in numeric_cand_cols if x not in linear_interpol_cols]
            
            if len(numeric_cand_cols)==0:
                return 
            

            # Interpolation test for polynomial of degree=2
            poly2_interpol_cols = []

            for cnum, col in enumerate(numeric_cand_cols):                
                print(f"\r2-polynomial interpolation test: {cnum+1}/{len(X_rem.columns)} columns processed", end="", flush=True)
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
            
            X_rem = X_rem.drop(poly2_interpol_cols,axis=1)
            numeric_cand_cols = [x for x in numeric_cand_cols if x not in poly2_interpol_cols]
            
            if len(numeric_cand_cols)==0:
                return 
            # Interpolation test for polynomial of degree=3
            poly3_interpol_cols = []

            for cnum, col in enumerate(numeric_cand_cols):                
                print(f"\r3-polynomial interpolation test: {cnum+1}/{len(X_rem.columns)} columns processed", end="", flush=True)
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
            X_rem = X_rem.drop(poly3_interpol_cols,axis=1)
            numeric_cand_cols = [x for x in numeric_cand_cols if x not in poly3_interpol_cols]
            
            if len(numeric_cand_cols)==0:
                return  

            # TODO: Test whether polynomials of a higher degree can be useful

            ### Combination test
            # TODO: Test whether using a sklearn decision tree can speed things up
            comb_num = []
            comb_cat = []
            comb_bothfine = []
            irrelevant_cols = []
            for cnum, col in enumerate(numeric_cand_cols):                
                print(f"\rCombination test: {cnum+1}/{len(X_rem.columns)} columns processed", end="", flush=True)
                x_use = X_rem[[col]].copy()
                params = base_params.copy()
                params["max_bin"] = X_rem[col].nunique() #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                params["max_depth"] = 2 #if mode=="cat" else 2
                params["n_estimators"] = X_rem[col].nunique() # 10000 #min(max(int(X[col].nunique()/4),1),100)

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
                        self.dtypes[col] = "indifferent" #self.orig_dtypes[col]
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
                params = base_params.copy()
                params["max_bin"] = X_rem[col].nunique() #min([10000, X_rem[col].nunique()]) #if mode=="cat" else 2
                params["max_depth"] = 2 #if mode=="cat" else 2
                params["n_estimators"] = X_rem[col].nunique() # 10000 #min(max(int(X[col].nunique()/4),1),100)

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

            return 
        else:
            return 




    def transform(self, X):
        for col in self.col_names:
            for col in self.cat_dtype_maps:
                X[col] = X[col].astype(str).fillna("nan").astype(self.cat_dtype_maps[col])

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == """__main__""":
    import pickle
    import openml
    collection_id = 457  # TabArena
    # collection_id = 334 # Tabular benchmark categorical classification
    # collection_id = 335 # Tabular benchmark categorical regression
    # collection_id = 336 # Tabular benchmark numerical regression
    # collection_id = 337 # Tabular benchmark numerical classification
    # collection_id = 379 # TabZilla-hard
    # Fetch the benchmark suite
    benchmark_suite = openml.study.get_suite(collection_id)
    tasks = benchmark_suite.data
    results = {}
    for num, tid in enumerate(tasks[:]):
        n_folds = 10
        
        data = openml.datasets.get_dataset(tid)
        # if data.name=="APSFailure":
        #     continue
        print(data.name)
        # if data.name!="superconductivity":
        #     continue
        X, _, _, _ = data.get_data()
        y = X[data.default_target_attribute]
        X = X.drop(columns=[data.default_target_attribute])
        
        if y.nunique()>10 and y.dtypes in [int, float]: # Attention: maternal health risk has 3 classes - need to handle that better
        # if y.nunique()>10: # Attention: maternal health risk has 3 classes - need to handle that better
            target_type = "regression"        
            y = pd.Series(MinMaxScaler().fit_transform(y.to_frame())[:,0], index=y.index,name=y.name)
        else:
            target_type = "binary"
            y = (y==y.value_counts().index[0]).astype(float)

        X.columns = [str(col).replace('[', '').replace(']', '').replace(':', '')
                                            .replace('<', '').replace('>', '')
                                            .replace('=', '').replace(',', '')
                                            .replace(' ', '_') for col in X.columns]
        

        dtypes = X.dtypes


        ######## Feature Type Detection        
        ftd = FeatureTypeDetector(target_type=target_type)
        start = time.time()
        ftd.fit(X,y, verbose=True)
        end = time.time()
        results[data.name] = {
            "dtypes": ftd.dtypes,
            "significances": ftd.significances,
            "scores": ftd.scores,
            "time": end - start,
        }

        print("DETECTED DTYPES:")
        print(pd.Series(ftd.dtypes).value_counts())

        if data.name not in ["poker-hand"]:
            ########## PERFORMANCE PRIOR DETECTION 
            X_prior = X.copy()
            X_prior.loc[:, pd.Series(X_prior.dtypes).apply(lambda x: x not in ["object", str, "category"])] = X_prior.loc[:, pd.Series(X_prior.dtypes).apply(lambda x: x not in ["object", str, "category"])].astype(float)
            obj_cols = X_prior.select_dtypes(include=['object']).columns
            for col in obj_cols:
                X_prior[col] = X_prior[col].astype('category')
            
            if target_type == "binary":
                scorer = roc_auc_score #lambda ytr, ypr: -log_loss(ytr, ypr) # roc_auc_score
                model_class = lgb.LGBMClassifier
                dummy = DummyClassifier(strategy='prior')
                linear =  UnivariateLogisticClassifier() #PolynomialLogisticClassifier(degree=1) #UnivariateLogisticClassifier() # LogisticRegression(solver='liblinear')
                poly_2 = PolynomialLogisticClassifier(degree=2)
                poly_3 = PolynomialLogisticClassifier(degree=3)
                poly_4 = PolynomialLogisticClassifier(degree=4)
                poly_5 = PolynomialLogisticClassifier(degree=5)
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
                model_class = lgb.LGBMRegressor
                dummy = DummyRegressor(strategy='mean')
                linear = UnivariateLinearRegressor()
                poly_2 = PolynomialRegressor(degree=2)
                poly_3 = PolynomialRegressor(degree=3)
                poly_4 = PolynomialRegressor(degree=4)
                poly_5 = PolynomialRegressor(degree=5)
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            cv_scores_with_early_stopping = make_cv_scores_with_early_stopping(target_type=target_type, scorer=scorer, cv=cv, early_stopping_rounds=20, vectorized=False)

            base_params = {
                "objective": "binary" if target_type=="binary" else "regression",
                "boosting_type": "gbdt",
                # "n_estimators": 10000,
                "verbosity": -1
            }
            model = model_class(**base_params)
            pipe = Pipeline([("model", model)])
            prior_perf = cv_scores_with_early_stopping(X_prior, y, pipe)


            ########## PERFORMANCE CSV comparison 
            # TODO: Make sure that CSV version only differs in losing nominal information for numeric columns
            X.to_csv(f"X_{data.name}_temp.csv", index=False)
            X_csv = pd.read_csv(f"X_{data.name}_temp.csv")
            obj_cols = X_csv.select_dtypes(include=['object']).columns
            for col in obj_cols:
                X_csv[col] = X_csv[col].astype('category')
            obj_cols = X_csv.select_dtypes(include=[int]).columns
            for col in obj_cols:
                X_csv[col] = X_csv[col].astype(float)

            if any(X_csv.dtypes!=dtypes):
                
                model = model_class(**base_params)
                pipe = Pipeline([("model", model)])
                csv_perf = cv_scores_with_early_stopping(X_csv, y, pipe)
            else:
                print("No changes in dtypes - skipping CSV performance evaluation")
                csv_perf = prior_perf

            os.remove(f"X_{data.name}_temp.csv")
            ########## PERFORMANCE POST DETECTION 
            if any(pd.Series(ftd.dtypes)!=X_prior.dtypes):
                X_post = ftd.transform(X_prior.copy())
                
                model = model_class(**base_params)
                pipe = Pipeline([("model", model)])
                post_perf = cv_scores_with_early_stopping(X_post, y, pipe)
            else:
                print("No changes in dtypes - skipping post-detection performance evaluation")
                post_perf = prior_perf

            ########## PERFORMANCE WITHOUT IRRELEVANT
            if any(pd.Series(ftd.dtypes)=="irrelevant"):
                assignments = pd.Series(ftd.dtypes)
                try:
                    X_rel = X_post.copy()
                except:
                    X_rel = X_prior.copy()
                X_rel = X_rel.drop(X_rel.columns[assignments=="irrelevant"], axis=1)
                
                model = model_class(**base_params)
                pipe = Pipeline([("model", model)])
                irrel_perf = cv_scores_with_early_stopping(X_rel, y, pipe)
            else:
                print("No irrelevant columns - skipping performance evaluation without irrelevant")
                irrel_perf = [np.nan]*n_folds

            print(f"Finished for {data.name}")
            print(f"CSV performance: {round(np.mean(csv_perf), 5)} ({round(np.std(csv_perf), 5)})")
            print(f"Prior performance: {round(np.mean(prior_perf), 5)} ({round(np.std(prior_perf), 5)})")
            print(f"Post performance: {round(np.mean(post_perf), 5)} ({round(np.std(post_perf), 5)})")
            print(f"Post drop-irrelevant performance: {round(np.mean(irrel_perf), 5)} ({round(np.std(irrel_perf), 5)})")
            print("##########################################")
        else:
            print(f"Skipping performance evaluation for {data.name}")
            prior_perf = [np.nan]*n_folds
            csv_perf = [np.nan]*n_folds
            post_perf = [np.nan]*n_folds
            irrel_perf = [np.nan]*n_folds

        results[data.name]["performance"] = {
            "prior": prior_perf,
            "csv": csv_perf,
            "post": post_perf,
            "post_drop_irrel": irrel_perf,
        }

    # with open("curr_res_tabarena_withperformance_noirrelevant_defaultLGB.pkl", 'wb') as file:
    #     pickle.dump(results, file)


