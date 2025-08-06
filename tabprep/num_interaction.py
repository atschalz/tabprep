from narwhals import col
import numpy as np
import pandas as pd
from itertools import combinations 
from tabprep.utils import make_cv_function
from tabprep.proxy_models import TargetMeanRegressor, TargetMeanClassifier
from itertools import combinations, product
import re
from tabprep.preprocessors import drop_highly_correlated_features

'''
Next steps:
- Move combination test to BasePreprocessor
- combination test as a method to assess the performance of interaction features
- Add interaction test to BasePreprocessor
'''

from tabprep.base_preprocessor import BasePreprocessor
class NumericalInteractionDetector(BasePreprocessor):
    def __init__(self, 
                 target_type, 
                n_folds=5, alpha=0.1, significance_method='wilcoxon', mvp_criterion='significance', mvp_max_cols_use=100, verbose=True,
                interaction_types: list = ['+', '-', '/', 'x'], # ['+', '-', '/', 'x', '&']
                use_mvp=True,
                corr_thresh = 0.95,
                select_n_candidates=None,
                max_filter=5000,
                 execution_mode='independent', # ['independent', 'reduce', 'expand']
                 max_order=2, num_operations='all',
                 scores: dict = None, cv_func=None,
                 min_cardinality=6,
                 max_base_interactions=2000,
                 apply_filters=True,
                 candidate_cols=None,
                 ):
        # TODO: Include possibility to select operators
        super().__init__(target_type=target_type, n_folds=n_folds, alpha=alpha, significance_method=significance_method, mvp_criterion=mvp_criterion, mvp_max_cols_use=mvp_max_cols_use, verbose=verbose)
        self.interaction_types = interaction_types
        self.use_mvp = use_mvp
        self.corr_thresh = corr_thresh
        self.select_n_candidates = select_n_candidates
        self.max_filter = max_filter
        self.execution_mode = execution_mode
        self.max_order = max_order
        self.num_operations = num_operations
        self.scores = scores
        self.cv_func = cv_func
        self.min_cardinality = min_cardinality
        self.max_base_interactions = max_base_interactions
        self.apply_filters = apply_filters
        self.candidate_cols = candidate_cols

        
        if self.target_type=='regression':
            self.target_model = TargetMeanRegressor()
        else:
            self.target_model = TargetMeanClassifier()
        
        if scores is None:
            self.scores = {}

        if cv_func is None:
            self.cv_func = make_cv_function(target_type=self.target_type, n_folds=5)

        self.significances = {}
        self.new_col_set = []

    def get_interaction_candidates_bottomup(self, 
                                        x, X_num, y, 
                                        col_method='random', # ['random', 'target_corr', 'feature_corr']
                                        interaction_method='random', # ['random', 'correlation']
                                        n_candidates=1, replace=False,
                                        seed=42
                                        ):
        np.random.seed(seed)
        if col_method == 'random':
            cols_to_use = np.random.choice(X_num.columns, np.min([len(X_num.columns),n_candidates]), replace=replace)
        elif col_method == 'target_corr':
            cols_to_use = X_num.corrwith(y).abs().sort_values(ascending=False).index[:n_candidates]
        elif col_method == 'feature_corr':
            cols_to_use = X_num.corrwith(x).abs().sort_values(ascending=False).index[:n_candidates]
        else:
            raise ValueError(f"Unknown col_method: {col_method}")

        X_int = pd.DataFrame()
        if interaction_method == 'random':
            interactions = np.random.choice(["x","/","+","-"], np.min([len(X_num.columns),n_candidates]), replace=True)
            for col, interaction in zip(cols_to_use, interactions):
                if interaction == "/":
                    X_int[f"{x.name}_{interaction}_{col}"] = (x / X_num[col].replace(0, np.nan)).values
                elif interaction == "x":
                    X_int[f"{x.name}_{interaction}_{col}"] = (x * X_num[col]).values
                elif interaction == "-":
                    X_int[f"{x.name}_{interaction}_{col}"] = (x - X_num[col]).values
                elif interaction == "+":
                    X_int[f"{x.name}_{interaction}_{col}"] = (x + X_num[col]).values
        elif interaction_method == 'correlation':
            X_int = pd.DataFrame(
                index=X_num.index, 
                columns=[f"{x.name}_/_{col2}" for col2 in cols_to_use]+ \
                [f"{x.name}_x_{col2}" for col2 in cols_to_use]+ \
                [f"{x.name}_-_{col2}" for col2 in cols_to_use]+ \
                [f"{x.name}_+_{col2}" for col2 in cols_to_use]
            )
            X_int[[f"{x.name}_/_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) / X_num[cols_to_use].replace(0, np.nan).values).values
            X_int[[f"{x.name}_x_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) * X_num[cols_to_use].values).values
            X_int[[f"{x.name}_-_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) - X_num[cols_to_use].values).values
            X_int[[f"{x.name}_+_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) + X_num[cols_to_use].values).values

            # Filter weird cases
            X_int = X_int.loc[:, X_int.nunique()>2] 

            cors = X_int.corrwith(y, method='spearman').abs()
            final_cols = [cors[[interaction for interaction in cors.index if col in interaction]].idxmax() for col in cols_to_use]

            X_int = X_int[final_cols]
        else:
            raise ValueError(f"Unknown interaction_method: {interaction_method}")
        
        return X_int

    def get_interaction_candidates_full(self, 
                                        x, X_num, y, 
                                        method='target_corr', # ['target_corr', 'corr_improve_over_base']
                                        n_candidates=1, 
                                        seed=42
                                        ):
        np.random.seed(seed)
        ### 1. Get Interactions
        cols_to_use = [c for c in X_num.columns if c != x.name]
        
        X_int = pd.DataFrame(
            index=X_num.index, 
            columns=[f"{x.name}_/_{col2}" for col2 in cols_to_use]+ \
            [f"{x.name}_x_{col2}" for col2 in cols_to_use]+ \
            [f"{x.name}_-_{col2}" for col2 in cols_to_use]+ \
            [f"{x.name}_+_{col2}" for col2 in cols_to_use]
        )
        
        # TODO: Think about whether reversing - and / makes sense
        # TODO: Think about whether setting nans for division by zero is appropriate
        X_int[[f"{x.name}_/_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) / X_num[cols_to_use].replace(0, np.nan).values).values
        X_int[[f"{x.name}_x_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) * X_num[cols_to_use].values).values
        X_int[[f"{x.name}_-_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) - X_num[cols_to_use].values).values
        X_int[[f"{x.name}_+_{col2}" for col2 in cols_to_use]] = (pd.concat([x]*len(cols_to_use),axis=1) + X_num[cols_to_use].values).values

        # Filter weird cases
        X_int = X_int.loc[:, X_int.nunique()>2]
        
        corr_X_int = X_int.corrwith(y, method='spearman').abs().sort_values(ascending=False)

        if n_candidates=='all':
            return X_int
        else:
            if method == 'target_corr':
                candidate_cols = corr_X_int.index[:n_candidates].tolist()
            # EXPERIMENTAL;DOESNT WORK CURRENTLY
            # elif method == 'corr_improve_over_base':
            #     base_cors = X_num.corrwith(y).abs()
            #     int_cors = corr_X_int.to_frame()
            #     int_cors_diff = int_cors.apply(lambda x: float([x-base_cors.loc[x.name.split(f"_{i}_")].max() for i in ['+', '-', '/', 'x'] if len(x.name.split(f"_{i}_"))>1][0][0]),axis=1)
            #     candidate_cols = int_cors_diff.sort_values(ascending=False).index[:1].tolist()
            else:
                raise ValueError(f"Unknown method: {method}. Use 'target_corr' or 'corr_improve_over_base'.")
            
            return X_int[candidate_cols]

    def get_interaction_candidates(self, x, X_num, y, interaction_mode='random', n_interaction_candidates=1):
        if interaction_mode=='full':
            X_int = self.get_interaction_candidates_full(x, X_num, y, method='target_corr', n_candidates=n_interaction_candidates)
        elif interaction_mode=='random':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='random', interaction_method='random',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='target':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='target_corr', interaction_method='random',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='feature':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='feature_corr', interaction_method='random',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='random-best':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='random', interaction_method='correlation',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='target-best':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='target_corr', interaction_method='correlation',
                                                        n_candidates=n_interaction_candidates, replace=False)
        elif interaction_mode=='feature-best':
            X_int = self.get_interaction_candidates_bottomup(x, X_num, y,
                                                        col_method='feature_corr', interaction_method='correlation',
                                                        n_candidates=n_interaction_candidates, replace=False)
        else:
            raise ValueError(f"Unknown interaction_mode: {interaction_mode}. Use 'full', 'random', 'target', 'feature', 'random-best', 'target-best', or 'feature-best'.")
        
        return X_int
    
    def interaction_test(self, X_cand, y, X_num):
        '''
        NOTE: This test is WIP and currently not used in the default pipeline.
        Three aspects matter:
        1. Does the new feature behaves numerically? - are the interpolation and combination tests positive?
        2. Does the feature improve performance over the base features at all?
        3. Is the feature interaction truly numerical or just the same as a combination of two base features?
        Therefore, we label a feature as numerical if its interaction 
            a) behaves numerically,
            b) improves performance, and 
            c) is not just a combination of the base features. 
        '''
       
        interaction_cols = []
        for cnum, col in enumerate(X_cand.columns):
            if self.verbose:
                print(f"\rInteraction test for column {cnum+1}/{len(X_cand.columns)} columns processed", end="", flush=True)

            # 1. Get interaction candidates
            X_int = self.get_interaction_candidates(X_cand[col], X_num, y,
                                                    interaction_mode=self.interaction_mode,
                                                    n_interaction_candidates=self.n_interaction_candidates)
   
            for highest_corr in X_int.columns:
                col2 = [highest_corr.split(f"_{i}_")[1] for i in ['+', '-', '/', 'x'] if len(highest_corr.split(f"_{i}_"))>1][0]
                X_use = X_int[[highest_corr]]
                arithmetic_col = X_use.columns[0]
                self.scores[arithmetic_col] = {}
                self.significances[arithmetic_col] = {}
                
                # base_performance = pd.Series({col: np.mean(self.scores[col]['mean']) for col in [col,col2]})
                # stronger_col = base_performance.idxmax()
                stronger_col, stronger_col_setting = pd.concat({col: pd.Series(self.scores[col]).apply(lambda x: x.mean()) for col in [col,col2]}).idxmax()

                ### 3. Get regular, binned and polynomial performance of the interaction feature and test significance
                # Regular TE performance
                self.get_dummy_mean_scores(X_use, y)
                self.significances[col][f"test_arithmetic-mean_superior_single-best"] = self.significance_test(
                    self.scores[arithmetic_col]['mean'] - self.scores[stronger_col][stronger_col_setting]
                )       
                if self.significances[col][f"test_arithmetic-mean_superior_single-best"]<self.alpha:
                    arithmetic_improves_performance = True
                else:
                    arithmetic_improves_performance = False

                ### Combination test
                self.combination_test(X_use, y, early_stopping=False, assign_dtypes=False, verbose=False)
                # 1. CHECK - The arithmetic combination feature behaves numerically
                arithmetic_is_numeric = False
                best_setting = "mean"
                m_bins = [2**i for i in range(1,100) if 2**i>= self.combination_test_min_bins and 2**i <= self.combination_test_max_bins]
                for m_bin in m_bins:
                    nunique = X_use[arithmetic_col].nunique()
                    if m_bin > nunique:
                        continue
                    if self.significances[arithmetic_col][f"test_combination_test_{m_bin}_superior"]<self.alpha:
                        arithmetic_is_numeric = True
                        best_setting = f"combination_test_{m_bin}"

                    self.significances[col][f"test_arithmetic-combination{m_bin}_superior_single-best"] = self.significance_test(
                        self.scores[arithmetic_col][f"combination_test_{m_bin}"] - self.scores[stronger_col][stronger_col_setting]
                    )       
                    if self.significances[col][f"test_arithmetic-combination{m_bin}_superior_single-best"]<self.alpha:
                        arithmetic_improves_performance = True

                    if arithmetic_is_numeric and arithmetic_improves_performance:
                        break
                

                # 2. CHECK - The arithmetic combination feature improves performance over the base features
                self.significances[col][f"test_arithmetic-best_superior_single-best"] = self.significance_test(
                    self.scores[arithmetic_col][best_setting] - self.scores[stronger_col][stronger_col_setting]
                )       
                if self.significances[col][f"test_arithmetic-best_superior_single-best"]<self.alpha:
                    arithmetic_improves_performance = True

                # 3. CHECK - The arithmetic combination feature is not just a combination of the base features

                if arithmetic_is_numeric and arithmetic_improves_performance:
                    self.dtypes[col] = "numeric"
                    interaction_cols.append(col)
                    break


                ### 5. Test whether categorical interaction is better than numerical interaction
                # Combine interaction performance
                # x = X_full[col].astype(str) + X_full[col2].astype(str)
                # if self.target_type == "regression":
                #     self.scores[arithmetic_col]['combine'] = self.cv_func(x.to_frame(), y, Pipeline(steps=[('model', TargetMeanRegressor())]))
                # else:
                #     self.scores[arithmetic_col]['combine'] = self.cv_func(x.to_frame(), y, Pipeline(steps=[('model', TargetMeanClassifier())]))
                

        remaining_cols = [x for x in X_cand.columns if x not in interaction_cols]

        return remaining_cols            
    

    def get_all_possible_interactions(self, X_num, order=2, max_base_interactions=10000):
        X = X_num.copy()
        X_int = X_num.copy()
        all_new_cols = np.array(list(combinations(X.columns,order)))
        np.random.shuffle(all_new_cols)
        feat0,feat1 = all_new_cols[:max_base_interactions].transpose()
        
        new_col_names = []
        for i_type in self.interaction_types:
            new_col_names += [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
        # X_int = pd.DataFrame(
            # index=X_num.index, 
            # columns=new_col_names
        # )
        new_cols = []
        for i_type in self.interaction_types:
            if i_type == '/':
                X_div = pd.DataFrame((X_num[feat0].values / X_num[feat1].replace(0, np.nan).values).astype(float))
                X_div.columns = [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
                new_cols.append(X_div)
            elif i_type == 'x':
                X_mult = pd.DataFrame((X_num[feat0].values * X_num[feat1].values).astype(float))
                X_mult.columns = [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
                new_cols.append(X_mult)
            elif i_type == '-':
                X_sub = pd.DataFrame((X_num[feat0].values - X_num[feat1].values).astype(float))
                X_sub.columns = [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
                new_cols.append(X_sub)
            elif i_type == '+':
                X_add = pd.DataFrame((X_num[feat0].values + X_num[feat1].values).astype(float))
                X_add.columns = [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
                new_cols.append(X_add)
            else:
                raise ValueError(f"Unknown interaction type: {i_type}. Use '/', 'x', '-', or '+'.")

        X_int = pd.concat(new_cols, axis=1)
        return X_int

    def add_higher_interaction(self, X_base_in, X_interact_in, max_base_interactions=10000):
        X_base = X_base_in.copy()
        X_interact = X_interact_in.copy()
        all_new_cols = np.array([[i, j] for i,j in product(X_interact.columns, X_base.columns) if j not in i])
        np.random.shuffle(all_new_cols)
        feat0,feat1 = all_new_cols[:max_base_interactions].transpose()
        
        new_col_names = []
        for i_type in self.interaction_types:
            new_col_names += [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
        # X_int = pd.DataFrame(
            # index=X_num.index, 
            # columns=new_col_names
        # )
        new_cols = []
        for i_type in self.interaction_types:
            if i_type == '/':
                X_div = pd.DataFrame((X_interact[feat0].values / X_base[feat1].replace(0, np.nan).values).astype(float))
                X_div.columns = [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
                new_cols.append(X_div)
            elif i_type == 'x':
                X_mult = pd.DataFrame((X_interact[feat0].values * X_base[feat1].values).astype(float))
                X_mult.columns = [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
                new_cols.append(X_mult)
            elif i_type == '-':
                X_sub = pd.DataFrame((X_interact[feat0].values - X_base[feat1].values).astype(float))
                X_sub.columns = [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
                new_cols.append(X_sub)
            elif i_type == '+':
                X_add = pd.DataFrame((X_interact[feat0].values + X_base[feat1].values).astype(float))
                X_add.columns = [f"{col1}_{i_type}_{col2}" for col1, col2 in zip(feat0, feat1)]
                new_cols.append(X_add)
            else:
                raise ValueError(f"Unknown interaction type: {i_type}. Use '/', 'x', '-', or '+'.")

        X_int_new = pd.concat(new_cols, axis=1)
        return pd.concat([X_interact, X_int_new], axis=1)

    def multivariate_performance_test(self, X_cand_in, y_in, 
                                      test_cols, max_cols_use=100):
        suffix = 'NUM-INT'
        self.new_col_set = super().multivariate_performance_test(X_cand_in, y_in, 
                                      test_cols, suffix=suffix,  max_cols_use=max_cols_use)
                
        print(f"{len(self.new_col_set)} new columns after multivariate performance test.")
        
        rejected_cols = set(test_cols) - set(self.new_col_set)
        return X_cand_in.drop(rejected_cols,axis=1)
    
    def remove_same_range_features(self, X, x):
        col = x.name
        feature_names = [f for f in re.split(r'_(x|/|\+|\-)_', col) if f not in {'x', '/', '+', '-'}]

        return X[feature_names].corrwith(x, method='spearman').max()

    def remove_constant_features(self, X):
        return X.loc[:, X.std() > 0]
    
    def remove_mostlynan_features(self, X):
        return X.loc[:, X.isna().mean() < 0.99]
    
    def fast_spearman(self, X):
        ranks = X.rank(axis=0)
        corrs = np.corrcoef(ranks.fillna(-1), rowvar=False)

        return pd.DataFrame(corrs, index=X.columns, columns=X.columns)

    def fit(self, X_in, y_in):        
        X = X_in.copy().reset_index(drop=True)
        y = y_in.copy().reset_index(drop=True)

        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(exclude=[np.number])

        if self.candidate_cols is not None:
            X_num = X_num[self.candidate_cols]

        # Shrink numerical features to the necessary ones
        X_num = X_num.loc[:, X_num.nunique() > self.min_cardinality]
        X_num = self.remove_mostlynan_features(X_num)
        # X_num = self.remove_constant_features(X_num)

        if len(X_num.columns) == 0:
            self.detection_attempted = False
            self.new_cols = []
            return self
        X_int = self.get_all_possible_interactions(X_num, order=2, max_base_interactions=self.max_base_interactions)
        
        # if self.apply_filters:
        #     X_all = pd.concat([X_num, X_int], axis=1)
        #     X_corr = self.fast_spearman(X_all)
        #     X_uncorr = drop_highly_correlated_features(X_corr)
        #     X_int = X_int.loc[:, X_int.columns.isin(X_uncorr.columns)]
        
        curr_order = 2
        max_candidates = self.select_n_candidates*(self.max_order-1)
        while X_int.shape[1] < max_candidates and X_num.shape[1] > curr_order:
            if curr_order == self.max_order:
                break
            X_int = self.add_higher_interaction(X_num, X_int, max_base_interactions=self.max_base_interactions)
            # if self.apply_filters:
            #     X_int = X_int[np.unique(X_int.columns)]
            #     X_corr = self.fast_spearman(pd.concat([X_num, X_int], axis=1))
            #     X_uncorr = drop_highly_correlated_features(X_corr)
            #     X_int = X_int.loc[:, X_int.columns.isin(X_uncorr.columns)]

            curr_order += 1
        cand_cols = list(set(X_int.columns) - set(X_num.columns))

        if len(cand_cols) > self.max_filter:
            X_int = X_int.sample(n=self.max_filter, random_state=42, replace=False, axis=1)
            
        # if self.apply_filters:
        # Filter constant features
        X_int = self.remove_constant_features(X_int)
        cand_cols = X_int.columns.tolist()
        
        # Filter features for which the range is too similar to the base features
        if self.select_n_candidates is not None:
            X_abs_corr = X_int.apply(lambda x: self.remove_same_range_features(X_num, x)).abs()
            cand_cols = X_abs_corr.sort_values().index[:self.select_n_candidates]
            X_int = X_int[cand_cols]
        elif self.corr_thresh<1:
            X_abs_corr = X_int.apply(lambda x: self.remove_same_range_features(X_num, x)).abs()
            cand_cols = X_int.columns[X_abs_corr < self.corr_thresh]
            X_int = X_int[cand_cols]

        if self.use_mvp:
            X_int = self.multivariate_performance_test(pd.concat([X, X_int], axis=1), y,
                                                    test_cols=cand_cols, max_cols_use=self.mvp_max_cols_use
                                                    )


        self.new_cols = X_int.columns.tolist()

        # if len(X_num.columns) == 0:
        #     raise ValueError("No numerical columns found in the dataset.")

        # # Iterate through numerical columns
        # for col in X_num.columns:
        #     x = X_num[col]
        #     if x.nunique() < self.min_cardinality:
        #         continue
            
        #     if self.execution_mode == 'independent':
        #         remaining_cols = self.interaction_test(x, y, X_num, )
        #     elif self.execution_mode == 'reduce':
        #         remaining_cols = self.interaction_test(x, y, X_num)
        #     elif self.execution_mode == 'expand':
        #         remaining_cols = self.interaction_test(x, y, X_num)
        #     else:
        #         raise ValueError(f"Unknown execution_mode: {self.execution_mode}. Use 'independent', 'reduce', or 'expand'.")

        #     # Update the new column set
        #     self.new_col_set.extend(remaining_cols)

        return self
    
    def transform(self, X):
        X_out = X.copy()
        new_cols = []
        for col in self.new_cols:
            feature_names = [f for f in re.split(r'_(x|/|\+|\-)_', col) if f not in {'x', '/', '+', '-'}]
            op_names = [f for f in re.split(r'_(x|/|\+|\-)_', col) if f in {'x', '/', '+', '-'}]
            # eval_str = "".join([f'X_out["{f}"]' if f not in {'x', '/', '+', '-'} else f for f in re.split(r'_(x|/|\+|\-)_', col)])
            new_feat = X_out[feature_names[0]].copy()
            for f, op in zip(feature_names[1:], op_names):
                if op == 'x':
                    new_feat *= X_out[f]
                elif op == '/':
                    new_feat /= X_out[f].replace(0, np.nan)
                elif op == '+':
                    new_feat += X_out[f]
                elif op == '-':
                    new_feat -= X_out[f]
                else:
                    raise ValueError(f"Unknown operator: {op}")
            new_feat.name = col
            new_cols.append(new_feat)
        X_out = pd.concat([X_out] + new_cols, axis=1)

        return X_out

if __name__ == "__main__":
    from ft_detection import clean_feature_names
    import os
    from tabprep.utils import *
    import openml
    from ft_detection import clean_feature_names
    benchmark = "TabArena"  # or "TabArena", "TabZilla", "Grinsztajn"
    dataset_name = 'coil'
    for benchmark in ['TabArena']: # ["Grinsztajn", "TabArena", "TabZilla"]:
        exp_name = f"EXP_num_interaction_{benchmark}"
        if os.path.exists(f"{exp_name}.pkl"):
            with open(f"{exp_name}.pkl", "rb") as f:
                results = pickle.load(f)
        else:
            results = {}
            results['performance'] = {}
            results['iterations'] = {}
            results['significances'] = {}

        tids, dids = get_benchmark_dataIDs(benchmark)  

        remaining_cols = {}

        for tid, did in zip(tids, dids):
            task = openml.tasks.get_task(tid)  # to check if the datasets are available
            data = openml.datasets.get_dataset(did)  # to check if the datasets are available
            if dataset_name not in data.name:
                continue
        
            
            if data.name in results['performance']:
                print(f"Skipping {data.name} as it already exists in results.")
                print(pd.DataFrame(results['performance'][data.name]).mean().sort_values(ascending=False))
                continue
            # else:
            #     break
            print(data.name)
            if data.name == 'guillermo':
                continue
            X, _, _, _ = data.get_data()
            y = X[data.default_target_attribute]
            X = X.drop(columns=[data.default_target_attribute])
            
            # X = X.sample(n=1000)
            # y = y.loc[X.index]

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
            
            detector = NumericalInteractionDetector(
                target_type=target_type, 
                max_order=2, num_operations='all',
                use_mvp=False,
                select_n_candidates=1000,
            )

            detector.fit(X, y)
            detector.transform(X)
            print(f"Found {len(detector.new_cols)} new interaction columns for {data.name}.")

            break
        break