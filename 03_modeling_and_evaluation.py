"""
Report pipeline step 03: Modeling and evaluation.

Purpose
-------
1. Compare tree-based models on the same Open / Click-after-Open tasks.
2. Keep the existing LightGBM + XAI pipeline from A_ml_modeling_v2.py as the final interpretation pipeline.

Notes
-----
- Logistic Regression is intentionally excluded.
- Random Forest and XGBoost use a common sklearn preprocessing pipeline.
- The original A_ml_modeling_v2.py file is not modified or deleted.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost may not be installed in the current environment.
    XGBClassifier = None

import A_ml_modeling_v2 as base


OUTPUT_DIR = base.OUTPUT_DIR
RANDOM_STATE = base.RANDOM_STATE


warnings.filterwarnings("ignore", category=FutureWarning)


def make_onehot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_common_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    """Build a common preprocessing pipeline for model comparison."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def build_model_candidates(y_train: pd.Series) -> dict[str, object]:
    """Create model candidates excluding Logistic Regression."""
    candidates: dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM_onehot": LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        ),
    }

    if XGBClassifier is not None:
        positive = int(np.sum(y_train))
        negative = int(len(y_train) - positive)
        scale_pos_weight = negative / positive if positive else 1.0
        candidates["XGBoost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        print("[Skip] XGBoost is not installed. Install xgboost to include it in comparison.")

    return candidates


def evaluate_candidate_model(
    task_name: str,
    model_name: str,
    estimator: object,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    categorical_features: list[str],
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one candidate and evaluate it with validation-selected threshold."""
    preprocessor = build_common_preprocessor(categorical_features, numeric_features)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    valid_pred = pipeline.predict_proba(X_valid)[:, 1]
    test_pred = pipeline.predict_proba(X_test)[:, 1]

    full_model_name = f"{task_name}_{model_name}"
    threshold_tuning = base.make_threshold_tuning_table(full_model_name, y_valid, valid_pred)
    threshold = base.select_best_threshold(threshold_tuning)
    threshold_tuning["selected"] = threshold_tuning["threshold"].eq(threshold)

    metric_rows = []
    for split, y_true, pred in [("valid", y_valid, valid_pred), ("test", y_test, test_pred)]:
        row = {
            "task": task_name,
            "model": model_name,
            "split": split,
        }
        row.update(base.evaluate_binary_classifier(y_true, pred, threshold))
        metric_rows.append(row)

    threshold_tuning.insert(0, "task", task_name)
    threshold_tuning["model_type"] = model_name
    return pd.DataFrame(metric_rows), threshold_tuning


def run_model_comparison_for_task(df: pd.DataFrame, target: str, task_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Random Forest, XGBoost, and one-hot LightGBM comparison for one task."""
    features, categorical_features, numeric_features = base.get_feature_columns()

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        train_df,
        valid_df,
        test_df,
    ) = base.split_by_time(df, target, features)

    split_summary = base.make_split_summary(train_df, valid_df, test_df, target)
    split_summary.insert(0, "task", task_name)

    metric_tables = []
    threshold_tables = []
    for model_name, estimator in build_model_candidates(y_train).items():
        print(f"   Train {task_name} - {model_name}")
        metrics, threshold_tuning = evaluate_candidate_model(
            task_name=task_name,
            model_name=model_name,
            estimator=estimator,
            X_train=X_train,
            X_valid=X_valid,
            X_test=X_test,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )
        metric_tables.append(metrics)
        threshold_tables.append(threshold_tuning)

    return (
        split_summary,
        pd.concat(metric_tables, ignore_index=True),
        pd.concat(threshold_tables, ignore_index=True),
    )


def run_model_comparison(df: pd.DataFrame) -> None:
    """Run model comparison and save report-friendly tables."""
    print("\n[1/2] Run model comparison")

    open_split, open_metrics, open_thresholds = run_model_comparison_for_task(
        df=df,
        target="target_opened",
        task_name="Open",
    )

    click_df = df[df["target_opened"].eq(1)].copy()
    click_split, click_metrics, click_thresholds = run_model_comparison_for_task(
        df=click_df,
        target="target_clicked",
        task_name="Click-after-Open",
    )

    split_summary = pd.concat([open_split, click_split], ignore_index=True)
    model_comparison = pd.concat([open_metrics, click_metrics], ignore_index=True)
    threshold_tuning = pd.concat([open_thresholds, click_thresholds], ignore_index=True)

    base.save_table(split_summary, "report_01_split_summary.csv")
    base.save_table(model_comparison, "report_02_model_comparison.csv")
    base.save_table(threshold_tuning, "report_03_threshold_tuning.csv")

    print("\n[Test model comparison]")
    print(model_comparison[model_comparison["split"].eq("test")].to_string(index=False))


def run_final_lightgbm_xai_pipeline() -> None:
    """Run the existing final LightGBM + XAI pipeline without changing the original file."""
    print("\n[2/2] Run final LightGBM + XAI pipeline from A_ml_modeling_v2.py")
    base.main()


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:.6f}".format)

    base.ensure_output_dir()

    print("Load final dataset")
    df = pd.read_parquet(base.DATA_PATH)
    print(f"   rows: {len(df):,}")
    print(f"   period: {df['sent_at'].min()} ~ {df['sent_at'].max()}")
    print(f"   open_rate: {df['target_opened'].mean():.6f}")
    print(f"   ctr: {df['target_clicked'].mean():.6f}")
    print(f"   ctor: {df.loc[df['target_opened'].eq(1), 'target_clicked'].mean():.6f}")

    run_model_comparison(df)
    run_final_lightgbm_xai_pipeline()

    print(f"\nSaved outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
