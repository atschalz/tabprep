import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from fe_utils import fe_combine
from utils import sample_from_set
from itertools import combinations
from utils import make_cv_scores_with_early_stopping, p_value_wilcoxon_greater_than_zero, TargetMeanRegressor, TargetMeanClassifier, \
    DummyClassifier, DummyRegressor
from itertools import product, combinations
from category_encoders import LeaveOneOutEncoder
from sklearn.metrics import roc_auc_score, log_loss, root_mean_squared_error

class CategoricalInteractionDetector(TransformerMixin, BaseEstimator):
    # TODO: Write a common class for different categorical feature preprocessing modules
    # TODO: Add memory estimation and function to process columns in chunks if memory is not enough
    def __init__(self, 
                 target_type, 
                 execution_mode='independent', # ['independent', 'reduce', 'expand']
                 max_order=2, num_operations='all',
                 scores: dict = None, cv_func=None,
                 min_cardinality=6,
                 ):
        self.target_type = target_type
        self.execution_mode = execution_mode
        self.max_order = max_order
        self.num_operations = num_operations
        self.scores = scores
        self.cv_func = cv_func
        self.min_cardinality = min_cardinality
        
        if self.target_type=='regression':
            self.target_model = TargetMeanRegressor()
        else:
            self.target_model = TargetMeanClassifier()
        
        if scores is None:
            self.scores = {}

        if cv_func is None:
            self.cv_func = make_cv_scores_with_early_stopping(target_type=self.target_type, n_folds=5)

        self.significances = {}
        self.new_col_set = []

    # def combine(self, X_in, order=2, num_operations='all', seed=42, **kwargs):
    #     # TODO: Implement as matrix operations to speed up the process
    #     X = X_in.copy()
    #     feat_combs = set(combinations(X.columns, order))
    #     features = {}

    #     if num_operations == "all":
    #         feat_combs_use = feat_combs
    #     else:
    #         feat_combs_use = sample_from_set(feat_combs, num_operations)

    #     for num, f_use in enumerate(feat_combs_use):
    #         # try:
    #         name = "_&_".join([str(i) for i in sorted(f_use)])
    #         features[name] = X[list(f_use)].astype(str).apply(lambda x: "_&_".join(map(str, x)), axis=1)

    #     return pd.DataFrame(features)

    def combine(self, X_in, order=2, num_operations='all', seed=42, **kwargs):
        # TODO: Implement as matrix operations to speed up the process
        X = X_in.copy()
        X = X.astype('U')
        feat_combs = set(combinations(np.unique(X.columns), order))

        if num_operations == "all":
            feat_combs_use = feat_combs
        else:
            feat_combs_use = sample_from_set(feat_combs, num_operations)
        feat_combs_use_arr = np.array(list(feat_combs_use)).transpose()

        new_names = ["_&_".join([str(i) for i in sorted(f_use)]) for f_use in feat_combs_use]

        features = X[feat_combs_use_arr[0]].values
        for num, arr in enumerate(feat_combs_use_arr[1:]):
            features += "_&_" + X[arr].values

        return pd.DataFrame(features, columns=new_names, index=X.index)
    
    def combine_predefined(self, X_in, comb_lst, **kwargs):
        X = X_in.copy()
        X = X.astype('U')
        feat_combs_use = [i.split("_&_") for i in comb_lst]
        feat_combs_use_arr = np.array(feat_combs_use).transpose()

        features = X[feat_combs_use_arr[0]].values
        for num, arr in enumerate(feat_combs_use_arr[1:]):
            features += "_&_" + X[arr].values

        return pd.DataFrame(features, columns=comb_lst, index=X.index)

    def leave_one_out_test(self, X, y):
        if self.target_type in ['binary', 'multiclass']:
            dummy = DummyClassifier(strategy='prior')
            metric = roc_auc_score
        elif self.target_type == 'regression':
            dummy = DummyRegressor(strategy='mean')
            metric = lambda x,y: -root_mean_squared_error(x, y)

        # X_use = X.copy()
        loo_cols = []
        self.loo_scores = {}
        for cnum, col in enumerate(X.columns):                
            print(f"\rLeave-One-Out test: {cnum+1}/{len(X.columns)} columns processed", end="", flush=True)
            if cnum==0:
                dummy_pred = dummy.fit(X[[col]], y).predict(X[[col]])
                dummy_score = metric(y, dummy_pred)

            loo_pred = LeaveOneOutEncoder().fit_transform(X[col].astype('category'), y)[col]
            self.loo_scores[col] = metric(y, loo_pred)

            if self.loo_scores[col] < dummy_score:
                loo_cols.append(col)

        remaining_cols = [x for x in X.columns if x not in loo_cols]

        return remaining_cols

    def expand_combine(self, X_int_in, X_base_in, num_operations='all', seed=42, **kwargs):
        X_int = X_int_in.copy()
        X_base = X_base_in.copy()
        X_int = X_int.astype('U')
        X_base = X_base.astype('U')
        feat_combs = [(base, interact) for base, interact in product(X_base.columns, X_int.columns) if base not in interact]
        new_names = ["_&_".join(sorted(int_col.split('_&_')+[base_col])) for base_col, int_col in feat_combs]
        
        feat_combs_unique = []
        new_names_unique = []
        for f,n in zip(feat_combs, new_names):
            if n not in new_names_unique:
                feat_combs_unique.append(f)
                new_names_unique.append(n)

        if num_operations == "all":
            feat_combs_use = feat_combs_unique
        else:
            feat_combs_use = sample_from_set(feat_combs_unique, num_operations)

        # TODO: Make sure the feature values are in the same order as the columns
        feat_combs_use_arr = np.array(list(feat_combs_use)).transpose()

        features = X_base[feat_combs_use_arr[0]].values + "_&_" + X_int[feat_combs_use_arr[1]].values
        df_features = pd.DataFrame(features, columns=new_names_unique, index=X_base.index)
        df_features = df_features.T.drop_duplicates().T
        return df_features

        # for num, (base_col, int_col) in enumerate(feat_combs_use):
        #     raw_cols = int_col.split('_&_')+[base_col]
        #     name = "_&_".join(sorted(int_col.split('_&_')+[base_col]))

        #     features[name] = pd.concat([X_int[int_col], X_base[base_col]], axis=1).astype(str).apply(lambda x: "_&_".join(map(str, x)), axis=1)

        # return pd.DataFrame(features)

    def find_interactions_expand(self, X, y=None):
        # TODO: Combine the three interaction functions into one
        for order in range(2,self.max_order+1):
            print(f"\rFind order {order} interactions", end='', flush=True)
            X_interact = self.combine(X, order=2, num_operations=self.num_operations)
            
            add_col_set = []
            for num, col in enumerate(X_interact.columns):
                self.significances[col] = {}
                print(f"\rProcessing column {num+1}/{len(X_interact.columns)}", end='', flush=True)
                base_cols = col.split('_&_')
                self.scores[col] = self.cv_func(X_interact[[col]], y, Pipeline([('model', self.target_model)]))

                for base_col in base_cols:
                    self.significances[col][base_col] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col] - self.scores[base_col]
                    )

                if all(np.array(list(self.significances[col].values())) < 0.05):
                    add_col_set.append(col)
                    self.new_col_set.append(col)
            
            X = pd.concat([X,X_interact[add_col_set]], axis=1)
            X = X.copy() # Avoid defragmentation issues

    def find_interactions_independent(self, X, y=None):
        X_new = X.copy()
        for order in range(2,self.max_order+1):
            print(f"\rFind order {order} interactions", end='', flush=True)
            X_interact = self.combine(X, order=order, num_operations=self.num_operations)
            
            add_col_set = []
            for num, col in enumerate(X_interact.columns):
                self.significances[col] = {}
                print(f"\rProcessing column {num+1}/{len(X_interact.columns)}", end='', flush=True)
                base_cols = col.split('_&_')
                self.scores[col] = self.cv_func(X_interact[[col]], y, Pipeline([('model', self.target_model)]))

                for base_col in base_cols:
                    self.significances[col][base_col] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col] - self.scores[base_col]
                    )

                if all(np.array(list(self.significances[col].values())) < 0.05):
                    add_col_set.append(col)
                    self.new_col_set.append(col)
            X_new = pd.concat([X_new,X_interact[add_col_set]], axis=1)
            X_new = X_new.copy() # Avoid defragmentation issues

    def find_interactions_reduce(self, X, y=None):
        X_base = X.copy()
        X_new = X.copy()
        if X.shape[1] < self.max_order:
            print(f"Warning: The number of columns ({X.shape[1]}) is less than the max order ({self.max_order}). Reducing the max order to {X_new.shape[1]}.")
            use_order = X.shape[1]
        else:
            use_order = self.max_order

        for order in range(2,use_order+1):
            print(f"\rFind order {order} interactions", end='', flush=True)
            if order == 2:
                X_interact = self.combine(X, order=2, num_operations=self.num_operations)
            else:
                X_interact = self.expand_combine(X_interact[add_col_set], X_base, num_operations=self.num_operations)
            
            loo_cols = self.leave_one_out_test(X_interact, y)
            X_interact = X_interact[loo_cols]

            add_col_set = []
            for num, col in enumerate(X_interact.columns):
                self.significances[col] = {}
                print(f"\rProcessing column {num+1}/{len(X_interact.columns)}", end='', flush=True)

                self.scores[col] = self.cv_func(X_interact[[col]], y, Pipeline([('model', self.target_model)]))

                base_cols = col.split('_&_')
                
                if order>2:
                    # TODO: Also add this logic to other modes
                    child_cols = set(combinations(base_cols, order-1))
                    base_cols += ['_&_'.join(i) for i in child_cols]
                for base_col in base_cols:
                    if not base_col in self.scores:
                        raw_cols = base_col.split('_&_')
                        self.scores[base_col] = self.cv_func(self.combine(X[raw_cols], order=len(raw_cols)), y, Pipeline([('model', self.target_model)]))

                    self.significances[col][base_col] = p_value_wilcoxon_greater_than_zero(
                        self.scores[col] - self.scores[base_col]
                    )

                if all(np.array(list(self.significances[col].values())) < 0.05):
                    add_col_set.append(col)
                    self.new_col_set.append(col)
            
            if len(add_col_set) == 0:
                print(f"\rNo new interactions found for order {order}. Stopping.")
                break
            else:
                # Make sure add_col_set is unique
                add_col_set = np.unique(["_&_".join(sorted(i.split("_&_"))) for i in add_col_set]).tolist()
                X_new = pd.concat([X_new,X_interact[add_col_set]], axis=1)
                X_new = X_new.copy() # Avoid defragmentation issues

    def fit(self, X_in, y=None):
        # TODO: Add logic to filter correlated new features early on
        # TODO: Add three different execution modes: sequential independent, sequential with performant subset, and sequential with lower order as new features
        # TODO: Add leave-one-out test to get rid of trivial features
        X = X_in.copy()

        print(f"Removing {X.shape[1] - X.loc[:, X.nunique() > self.min_cardinality].shape[1]} low cardinality features with less than {self.min_cardinality} unique values")
        X = X.loc[:, X.nunique() > self.min_cardinality]  # Remove low cardinality features


        for col in X.columns:
            if col not in self.scores:
                self.scores[col] = self.cv_func(X[[col]], y, Pipeline([('model', self.target_model)]))
        
        if self.execution_mode == "expand":
            self.find_interactions_expand(X, y)
        elif self.execution_mode == "independent":
            self.find_interactions_independent(X, y)
        elif self.execution_mode == "reduce":
            self.find_interactions_reduce(X, y)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}. Use 'sequential' or 'independent'.")
        return self
    
    def transform(self, X_in):        
        X = X_in.copy()
        X = X.astype('U')

        X_out = X.copy()
        for degree in range(2, self.max_order+1):
            col_set_use = [col for col in self.new_col_set if col.count('_&_')+1 == degree]
            if len(col_set_use) > 0:
                X_degree = self.combine_predefined(X, col_set_use)
                X_out = pd.concat([X_out, X_degree], axis=1)

        return X_out
        # for num_new, col in enumerate(self.new_col_set):
        #     print(f"\rTransforming column {num_new+1}/{len(self.new_col_set)}\r", end='')
        #     base_cols = col.split('_&_')
        #     for num, base_col in enumerate(base_cols):
        #         if num == 0:
        #             X[col] = X[base_col]
        #         else:
        #             X[col] = X[col] + "_&_" + X[base_col]
        #     X = X.copy()
        # return X


if __name__ == "__main__":
    import openml
    import pandas as pd
    from sympy import rem
    from utils import get_benchmark_dataIDs, get_metadata_df
    from ft_detection import FeatureTypeDetector

    benchmark = "TabZilla"  # or "TabArena", "TabZilla", "Grinsztajn"
    # dataset_name = 'concrete_compressive_strength'  # (['airfoil_self_noise', 'Amazon_employee_access', 'anneal', 'Another-Dataset-on-used-Fiat-500', 'bank-marketing', 'Bank_Customer_Churn', 'blood-transfusion-service-center', 'churn', 'coil2000_insurance_policies', 'concrete_compressive_strength', 'credit-g', 'credit_card_clients_default', 'customer_satisfaction_in_airline', 'diabetes', 'Diabetes130US', 'diamonds', 'E-CommereShippingData', 'Fitness_Club', 'Food_Delivery_Time', 'GiveMeSomeCredit', 'hazelnut-spread-contaminant-detection', 'healthcare_insurance_expenses', 'heloc', 'hiva_agnostic', 'houses', 'HR_Analytics_Job_Change_of_Data_Scientists', 'in_vehicle_coupon_recommendation', 'Is-this-a-good-customer', 'kddcup09_appetency', 'Marketing_Campaign', 'maternal_health_risk', 'miami_housing', 'NATICUSdroid', 'online_shoppers_intention', 'physiochemical_protein', 'polish_companies_bankruptcy', 'APSFailure', 'Bioresponse', 'qsar-biodeg', 'QSAR-TID-11', 'QSAR_fish_toxicity', 'SDSS17', 'seismic-bumps', 'splice', 'students_dropout_and_academic_success', 'taiwanese_bankruptcy_prediction', 'website_phishing', 'wine_quality', 'MIC', 'jm1', 'superconductivity']) 

    tids, dids = get_benchmark_dataIDs(benchmark)  

    remaining_cols = {}

    for tid, did in zip(tids, dids):
        task = openml.tasks.get_task(tid)  # to check if the datasets are available
        data = openml.datasets.get_dataset(did)  # to check if the datasets are available
        # if data.name!=dataset_name:
        #     continue
        # else:
        #     break
        print(data.name)
        X, _, _, _ = data.get_data()
        y = X[data.default_target_attribute]
        X = X.drop(columns=[data.default_target_attribute])
        
        if benchmark == "Grinsztajn" and X.shape[0]>10000:
            X = X.sample(10000, random_state=0)
            y = y.loc[X.index]

        if task.task_type == "Supervised Classification":
            target_type = "binary" if y.nunique() == 2 else "multiclass"
        else:
            target_type = 'regression'
        if target_type=="multiclass":
            # TODO: Fix this hack
            y = (y==y.value_counts().index[0]).astype(int)  # make it binary
            target_type = "binary"
        elif target_type=="binary" and y.dtype not in ["int", "float", "bool"]:
            y = (y==y.value_counts().index[0]).astype(int)  # make it numeric
        else:
            y = y.astype(float)


        X_cat = X.select_dtypes(include=['object', 'category'])
        X_cat = X_cat.loc[:, X_cat.nunique() >2] # remove columns with less than 3 unique values
        print(f"Dataset: {data.name} with {X_cat.shape[1]} categorical columns")
        if X_cat.shape[1] > 1:
            cd = CategoricalInteractionDetector(target_type, max_order=3, execution_mode='reduce').fit(X_cat, y)
            X_new = cd.transform(X_cat)
            print(pd.DataFrame(cd.scores).mean().sort_values(ascending=False))
            print(f"New columns: {len(cd.new_col_set)}: {cd.new_col_set}")


        # detector = FeatureTypeDetector(target_type=target_type, 
        #                             interpolation_criterion="match",  # 'win' or 'match'
        #                             lgb_model_type='huge-capacity',
        #                             verbose=False)
        
        # # detector.fit(X, y, verbose=False)
        # # print(pd.Series(detector.dtypes).value_counts())
        
        # rem_cols: list = detector.handle_trivial_features(X)
        # print('--'*50)
        # print(f"\rTrivial features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue

        # X_num = X[rem_cols].astype(float).copy()
        
        # rem_cols = detector.get_dummy_mean_scores(X, y)
        # print('--'*50)
        # print(f"\rIrrelevant features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue    

        # rem_cols = detector.leave_one_out_test(X, y)
        # print('--'*50)
        # print(f"\rLeave-One-Out features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue

        
        # rem_cols = detector.combination_test(X, y, max_binning_configs=3)
        # print('--'*50)
        # print(f"\rCombination features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue
        
        # rem_cols = detector.interpolation_test(X, y, max_degree=3)
        # print('--'*50)
        # print(f"\rInterpolation features removed: {X.shape[1]-len(rem_cols)}\r")
        # X = X[rem_cols]
        # if len(rem_cols) == 0:
        #     remaining_cols[data.name] = []
        #     continue

        # rem_cols = detector.interaction_test(X, X_num, y)
        # print('--'*50)
        # print(f"\rInteraction features removed: {X.shape[1]-len(rem_cols)}\r")

        # # rem_cols = detector.performance_test(X, y)
        # # print('--'*50)
        # # print(f"\rLGB-performance features removed: {X.shape[1]-len(rem_cols)}\r")
        # # X = X[rem_cols]
        # # if len(rem_cols) == 0:
        # #     remaining_cols[data.name] = []
        # #     continue

        # remaining_cols[data.name] = rem_cols
        # print(f"{data.name} ({len(rem_cols)}): {rem_cols}")

        # # if len(rem_cols) > 0:
        # #     cat_res = CatResolutionDetector(target_type='regression').fit(X, y)
        # #     print(pd.Series(cat_res.optimal_thresholds).sort_values(ascending=False))

        # # if data.name=='nyc-taxi-green-dec-2016':
        # #     break

        # # for i, col in enumerate(X.columns):
        # #     grouped_interpolation_test(X[col],(y==y[0]).astype(int), 'binary', add_dummy=True if i==0 else False)




