"""This script reproduces the notebook's raw-vs-TabPrep sklearn comparisons:
- LogisticRegression
- RandomForestClassifier

Run from the `tabprep/` directory with:
`python example_sklearn.py`
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def make_preprocessor(columns):
    num_cols, cat_cols = columns
    return ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def main() -> None:
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper
    from tabprep import TabPrepFeatureGenerator

    task = OpenMLTaskWrapper.from_task_id(task_id=363618)
    problem_type = task.problem_type
    train_data, test_data = task.get_train_test_split_combined(fold=0)
    X = train_data.drop(columns=[task.label])
    y = train_data[task.label]
    X_test = test_data.drop(columns=[task.label])
    y_test = test_data[task.label]

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y), index=y.index)
    y_test = pd.Series(label_encoder.transform(y_test), index=y_test.index)

    print(f"Task: {task.task_id} ({task.label})")
    print(f"Problem type: {problem_type}")
    print(f"Raw dataset shape: {X.shape}")

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Numeric columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")

    tabprep_feature_generator = TabPrepFeatureGenerator(target_type="binary")
    X_train_tabprep = tabprep_feature_generator.fit_transform(X, y)
    X_test_tabprep = tabprep_feature_generator.transform(X_test)

    tabprep_num_cols = X_train_tabprep.select_dtypes(include=["number"]).columns.tolist()
    tabprep_cat_cols = X_train_tabprep.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"TabPrep numeric columns: {len(tabprep_num_cols)}")
    print(f"TabPrep categorical columns: {len(tabprep_cat_cols)}")
    assert len(tabprep_num_cols) + len(tabprep_cat_cols) == X_train_tabprep.shape[1]

    def fit_and_score(model, X_train, y_train, X_eval, y_eval) -> float:
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_eval)[:, 1]
        return roc_auc_score(y_eval, pred)

    raw_logreg = make_pipeline(
        make_preprocessor((num_cols, cat_cols)),
        LogisticRegression(max_iter=1000, random_state=7),
    )
    raw_logreg_score = fit_and_score(raw_logreg, X, y, X_test, y_test)

    tabprep_logreg = make_pipeline(
        make_preprocessor((tabprep_num_cols, tabprep_cat_cols)),
        LogisticRegression(max_iter=1000, random_state=7),
    )
    tabprep_logreg_score = fit_and_score(tabprep_logreg, X_train_tabprep, y, X_test_tabprep, y_test)

    raw_rf = make_pipeline(
        make_preprocessor((num_cols, cat_cols)),
        RandomForestClassifier(random_state=0),
    )
    raw_rf_score = fit_and_score(raw_rf, X, y, X_test, y_test)

    tabprep_rf = make_pipeline(
        make_preprocessor((tabprep_num_cols, tabprep_cat_cols)),
        RandomForestClassifier(random_state=0),
    )
    tabprep_rf_score = fit_and_score(tabprep_rf, X_train_tabprep, y, X_test_tabprep, y_test)

    sklearn_comparison = pd.DataFrame(
        [
            {
                "model_family": "raw sklearn + logistic regression",
                "roc_auc": raw_logreg_score,
                "n_features_train": X.shape[1],
                "n_features_test": X_test.shape[1],
            },
            {
                "model_family": "tabprep + logistic regression",
                "roc_auc": tabprep_logreg_score,
                "n_features_train": X_train_tabprep.shape[1],
                "n_features_test": X_test_tabprep.shape[1],
            },
            {
                "model_family": "raw sklearn + random forest",
                "roc_auc": raw_rf_score,
                "n_features_train": X.shape[1],
                "n_features_test": X_test.shape[1],
            },
            {
                "model_family": "tabprep + random forest",
                "roc_auc": tabprep_rf_score,
                "n_features_train": X_train_tabprep.shape[1],
                "n_features_test": X_test_tabprep.shape[1],
            },
        ]
    ).sort_values("roc_auc", ascending=False)

    print("\nSklearn comparison:")
    print(sklearn_comparison.to_string(index=False))

    return {
        "task": task,
        "X_train_tabprep": X_train_tabprep,
        "X_test_tabprep": X_test_tabprep,
        "sklearn_comparison": sklearn_comparison,
    }


if __name__ == "__main__":
    main()
