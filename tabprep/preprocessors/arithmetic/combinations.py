import numpy as np
import pandas as pd
from itertools import combinations, product
from math import comb

def get_all_bivariate_interactions(
    X_num, 
    order=2, 
    max_base_interactions=10000,
    interaction_types=['/', '*', '-', '+'], # TODO: make inverse_div an own op
    random_state=None
):
    rng = np.random.default_rng(random_state)

    cols = X_num.columns.to_numpy()
    combs = np.array(list(combinations(cols, order)))

    # Sample combinations directly instead of shuffling entire array
    if len(combs) > max_base_interactions:
        combs = combs[rng.choice(len(combs), max_base_interactions, replace=False)]

    feat0, feat1 = combs.T
    results = []

    # Pre-cache arrays for speed
    arr = X_num.to_numpy()
    col_idx = {c: i for i, c in enumerate(cols)}

    # Helper to quickly extract columns
    def get_pair_arrays(f0, f1):
        return arr[:, [col_idx[c] for c in f0]], arr[:, [col_idx[c] for c in f1]]

    for op in interaction_types:
        A, B = get_pair_arrays(feat0, feat1)
        # Avoid division by zero with masking instead of replace()
        with np.errstate(divide='ignore', invalid='ignore', over="ignore"):
            if op == '/':
                div1 = A / np.where(B == 0, np.nan, B)
                div2 = B / np.where(A == 0, np.nan, A)
                df1 = pd.DataFrame(div1, columns=[f"{a}_/_{b}" for a, b in zip(feat0, feat1)])
                df2 = pd.DataFrame(div2, columns=[f"{b}_/_{a}" for a, b in zip(feat0, feat1)])
                results.extend([df1, df2])
            elif op == '*':
                df = pd.DataFrame(A * B, columns=[f"{a}_*_{b}" for a, b in zip(feat0, feat1)])
                results.append(df)
            elif op == '-':
                df = pd.DataFrame(A - B, columns=[f"{a}_-_{b}" for a, b in zip(feat0, feat1)])
                results.append(df)
            elif op == '+':
                df = pd.DataFrame(A + B, columns=[f"{a}_+_{b}" for a, b in zip(feat0, feat1)])
                results.append(df)
            else:
                raise ValueError(f"Unknown operator '{op}'.")

    return pd.concat(results, axis=1)

def get_n_possible_interactions(n, order=2):
    return comb(n, order)

def add_higher_interaction(
        X_base, 
        X_interact, 
        max_base_interactions=10000,
        interaction_types=['/', '*', '-', '+'], # FIXME: Might need to fix bug if one of these operators occurs in feature names
        random_state=None
    ):
    rng = np.random.default_rng(random_state)

    # Generate valid column pairs (avoid j inside i for safety)
    all_pairs = [
        (i, j) for i, j in product(X_interact.columns, X_base.columns)
        if j not in i
    ]
    all_pairs = np.array(all_pairs)
    if len(all_pairs) > max_base_interactions:
        all_pairs = all_pairs[rng.choice(len(all_pairs), max_base_interactions, replace=False)]
    feat0, feat1 = all_pairs.T

    # Convert to numpy arrays once for speed
    X_interact_vals = X_interact[feat0].to_numpy(dtype=float)
    X_base_vals = X_base[feat1].to_numpy(dtype=float)

    new_data = {}

    for i_type in interaction_types:
        # Forward and reverse divisions
        with np.errstate(divide='ignore', invalid='ignore', over="ignore"):
            if i_type == '/':
                res1 = X_interact_vals / np.where(X_base_vals == 0, np.nan, X_base_vals)
                res2 = X_base_vals / np.where(X_interact_vals == 0, np.nan, X_interact_vals)
                names1 = [f"{a}_{i_type}_{b}" for a, b in zip(feat0, feat1)]
                names2 = [f"{b}_{i_type}_{a}" for a, b in zip(feat0, feat1)]
                new_data.update(dict(zip(names1, res1.T)))
                new_data.update(dict(zip(names2, res2.T)))

            elif i_type == '*':
                # Multiplication is commutative → skip duplicate (a×b == b×a)
                seen_pairs = set()
                res_list, name_list = [], []
                for a, b in zip(feat0, feat1):
                    if tuple(sorted((a, b))) in seen_pairs:
                        continue
                    seen_pairs.add(tuple(sorted((a, b))))
                    res_list.append((X_interact[a].values * X_base[b].values).astype(float))
                    name_list.append(f"{a}_{i_type}_{b}")
                if res_list:
                    res_arr = np.column_stack(res_list)
                    new_data.update(dict(zip(name_list, res_arr.T)))

            elif i_type == '+':
                # Addition is commutative → skip duplicate (a+b == b+a)
                seen_pairs = set()
                res_list, name_list = [], []
                for a, b in zip(feat0, feat1):
                    if tuple(sorted((a, b))) in seen_pairs: # FIXME: a can already consist of multiple features
                        continue
                    seen_pairs.add(tuple(sorted((a, b))))
                    res_list.append((X_interact[a].values + X_base[b].values).astype(float))
                    name_list.append(f"{a}_{i_type}_{b}")
                if res_list:
                    res_arr = np.column_stack(res_list)
                    new_data.update(dict(zip(name_list, res_arr.T)))

            elif i_type == '-':
                # Subtraction (non-commutative): only one direction (A−B)
                res = X_interact_vals - X_base_vals
                names = [f"{a}_{i_type}_{b}" for a, b in zip(feat0, feat1)]
                new_data.update(dict(zip(names, res.T)))

            else:
                raise ValueError(f"Unknown interaction type: {i_type}. Use '/', '*', '-', or '+'.")

    # Build the new DataFrame once (fast)
    X_int_new = pd.DataFrame(new_data, index=X_interact.index)

    # Return combined features
    return X_int_new #pd.concat([X_interact, X_int_new], axis=1)


def get_n_possible_higher_interactions(n_base, n_int):
    # Calculate the number of possible interactions
    return n_base * n_int * 5  # 5 for the different operations
