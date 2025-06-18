from ft_detection import FeatureTypeDetector
import pickle
import openml
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, log_loss, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from utils import make_cv_scores_with_early_stopping
import os


def get_benchmark_dataIDs(benchmark_name):
    if benchmark_name == "Grinstajn": # Use the most original version of each unique dataset
        # collection_id = 334 # Tabular benchmark categorical classification
        # collection_id = 335 # Tabular benchmark categorical regression
        # collection_id = 336 # Tabular benchmark numerical regression
        # collection_id = 337 # Tabular benchmark numerical classification
        joint_tasks = []
        unique_names = set()
        for collection_id in [335, 336,  334, 337]:
            benchmark_suite = openml.study.get_suite(collection_id)
            tasks = benchmark_suite.data
            for tid in tasks:
                data = openml.datasets.get_dataset(tid) 
                if data.name not in unique_names:
                    joint_tasks.append(tid)
                    unique_names.add(data.name)

        return joint_tasks
        
    elif benchmark_name == "TabArena":
        collection_id = 457  # TabArena
        benchmark_suite = openml.study.get_suite(collection_id)
        tasks = benchmark_suite.data
        return tasks

    elif benchmark_name == "TabZilla":
        collection_id = 379
        benchmark_suite = openml.study.get_suite(collection_id)
        tasks = benchmark_suite.data
        return tasks

if __name__ == """__main__""":
    benchmark = "Grinstajn" # "TabArena", "TabZilla", "Grinstajn"
    save_name = f"curr_res_{benchmark}.pkl"
    
    #############
    results = {}    
    tasks = get_benchmark_dataIDs(benchmark)
    
    for num, tid in enumerate(tasks[:]):
        n_folds = 10
        
        data = openml.datasets.get_dataset(tid)
        print(data.name)
        
        X, _, _, _ = data.get_data()
        y = X[data.default_target_attribute]
        X = X.drop(columns=[data.default_target_attribute])


        # TODO: Remove this hack and use the correct tasks instead
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
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                scorer = lambda ytr, ypr: -root_mean_squared_error(ytr, ypr) # r2_score
                model_class = lgb.LGBMRegressor
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

    with open(f"{save_name}.pkl", 'wb') as file:
        pickle.dump(results, file)


