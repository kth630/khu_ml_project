"""모델 성능 비교와 LightGBM 기반 XAI 산출물을 생성하는 스크립트.

실행 목적
---------
1. Open 예측과 Click-after-Open 예측에 대해 RandomForest, LightGBM, XGBoost를 비교한다.
2. 예측 성능 best model은 test PR-AUC 기준으로 선택한다.
3. 해석(XAI)은 LightGBM 모델만 대상으로 permutation importance와 SHAP을 계산한다.

주의
----
- 전처리 결과 파일(outputs/processed/A_ml_dataset.pkl)이 먼저 생성되어 있어야 한다.
- LightGBM은 범주형 변수를 native categorical 방식으로 사용한다.
- RandomForest와 XGBoost는 One-Hot Encoding pipeline을 사용한다.
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# 모든 입력/출력 경로는 email_ml 폴더를 기준으로 잡는다.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "outputs" / "processed" / "A_ml_dataset.pkl"
MODELS_DIR = ROOT / "outputs" / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"

RANDOM_STATE = 42
PERMUTATION_REPEATS = 3
PERMUTATION_SAMPLE_SIZE = 100_000
SHAP_SAMPLE_SIZE = 100_000
PREDICTION_SAMPLE_SIZE = 50_000
THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dirs() -> None:
    """모델과 성능 지표를 저장할 폴더를 생성한다."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_columns() -> tuple[list[str], list[str], list[str]]:
    """모델에 사용할 전체 feature, 범주형 feature, 수치형 feature 목록을 반환한다."""
    categorical_features = [
        "topic",
        "provider_group",
        "send_hour",
        "send_dow",
    ]

    numeric_features = [
        "days_since_last_email",
        "days_since_last_open",
        "days_since_last_click",
        "has_prior_email",
        "has_prior_open",
        "has_prior_click",
        "email_count_7d",
        "open_count_7d",
        "click_count_7d",
    ]

    return categorical_features + numeric_features, categorical_features, numeric_features


def get_feature_groups() -> dict[str, list[str]]:
    """permutation importance와 group SHAP에서 사용할 feature group을 정의한다."""
    return {
        "campaign_topic": ["topic"],
        "provider": ["provider_group"],
        "send_time": ["send_hour", "send_dow"],
        "history_recency": [
            "days_since_last_email",
            "days_since_last_open",
            "days_since_last_click",
        ],
        "history_prior": [
            "has_prior_email",
            "has_prior_open",
            "has_prior_click",
        ],
        "history_frequency": [
            "email_count_7d",
            "open_count_7d",
            "click_count_7d",
        ],
    }


def make_onehot_encoder() -> OneHotEncoder:
    """scikit-learn 버전에 맞는 OneHotEncoder를 생성한다."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_onehot_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    """RandomForest/XGBoost용 전처리 pipeline을 만든다."""
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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


def prepare_lgbm_categories(frames: list[pd.DataFrame], categorical_features: list[str]) -> list[pd.DataFrame]:
    """LightGBM native categorical 학습을 위해 범주형 컬럼을 pandas Categorical로 변환한다."""
    category_levels: dict[str, list[str]] = {}
    for col in categorical_features:
        levels: list[str] = []
        for frame in frames:
            levels.extend(frame[col].dropna().astype(str).unique().tolist())
        category_levels[col] = sorted(set(levels))

    prepared = []
    for frame in frames:
        out = frame.copy()
        for col in categorical_features:
            out[col] = pd.Categorical(out[col].astype(str), categories=category_levels[col])
        prepared.append(out)
    return prepared


def split_by_time(
    df: pd.DataFrame,
    target: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """sent_at 시간 순서 기준으로 train/valid/test를 60/20/20으로 분리한다."""
    ordered = df.sort_values("sent_at").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.6)
    valid_end = int(n * 0.8)

    train_df = ordered.iloc[:train_end].copy()
    valid_df = ordered.iloc[train_end:valid_end].copy()
    test_df = ordered.iloc[valid_end:].copy()

    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[target].astype(int).copy()
    y_valid = valid_df[target].astype(int).copy()
    y_test = test_df[target].astype(int).copy()

    return X_train, X_valid, X_test, y_train, y_valid, y_test, train_df, valid_df, test_df


def make_split_summary(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """split별 기간, 행 수, positive rate를 요약한다."""
    rows = []
    for split, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        rows.append(
            {
                "split": split,
                "rows": int(len(part)),
                "start_at": part["sent_at"].min(),
                "end_at": part["sent_at"].max(),
                "positive_rate": float(part[target].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_binary_classifier(y_true: pd.Series, pred_proba: np.ndarray, threshold: float) -> dict[str, float]:
    """threshold 기준으로 binary classification 지표를 계산한다."""
    y_pred = (np.asarray(pred_proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, pred_proba)),
        "pr_auc": float(average_precision_score(y_true, pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positive_rate": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(y_pred)),
    }


def make_threshold_tuning_table(task: str, model: str, y_true: pd.Series, pred_proba: np.ndarray) -> pd.DataFrame:
    """validation set에서 threshold 후보별 precision/recall/F1을 계산한다."""
    y_true_arr = np.asarray(y_true).astype(int)
    pred_proba_arr = np.asarray(pred_proba)
    base_positive_rate = float(np.mean(y_true_arr))

    rows = []
    for threshold in THRESHOLD_GRID:
        y_pred = (pred_proba_arr >= threshold).astype(int)
        rows.append(
            {
                "task": task,
                "model": model,
                "split": "valid",
                "threshold": float(threshold),
                "validation_precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
                "validation_recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
                "validation_f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
                "pred_positive_rate": float(np.mean(y_pred)),
                "base_positive_rate": base_positive_rate,
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(threshold_tuning: pd.DataFrame) -> float:
    """validation F1이 가장 높은 threshold를 선택한다."""
    selected = threshold_tuning.sort_values(
        ["validation_f1", "validation_precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(selected["threshold"])


def build_model_candidates(y_train: pd.Series) -> dict[str, object]:
    """성능 비교에 사용할 RandomForest와 XGBoost 모델을 생성한다."""
    candidates: dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
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
        print("[Skip] XGBoost is not installed.")

    return candidates


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, categorical_features: list[str]) -> LGBMClassifier:
    """native categorical LightGBM 모델을 학습한다."""
    model = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_features)
    return model


def safe_task_name(task: str) -> str:
    return task.lower().replace("-", "_").replace(" ", "_")


def safe_model_name(model: str) -> str:
    return model.lower().replace("-", "_").replace(" ", "_")


def evaluate_sklearn_candidate(
    task: str,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-Hot Encoding pipeline 기반 모델을 학습하고 평가한다."""
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_onehot_preprocessor(categorical_features, numeric_features)),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    valid_pred = pipeline.predict_proba(X_valid)[:, 1]
    test_pred = pipeline.predict_proba(X_test)[:, 1]

    threshold_tuning = make_threshold_tuning_table(task, model_name, y_valid, valid_pred)
    threshold = select_best_threshold(threshold_tuning)
    threshold_tuning["selected"] = threshold_tuning["threshold"].eq(threshold)

    model_path = MODELS_DIR / f"{safe_task_name(task)}_{safe_model_name(model_name)}.joblib"
    joblib.dump(pipeline, model_path)

    metrics = make_metric_rows(task, model_name, model_path, y_valid, y_test, valid_pred, test_pred, threshold)
    predictions = make_prediction_sample(task, model_name, y_test, test_pred, threshold)
    return metrics, threshold_tuning, predictions


def evaluate_lgbm_candidate(
    task: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    categorical_features: list[str],
) -> tuple[LGBMClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """LightGBM 모델을 학습하고 성능 평가 결과를 반환한다."""
    X_train_lgbm, X_valid_lgbm, X_test_lgbm = prepare_lgbm_categories(
        [X_train, X_valid, X_test],
        categorical_features,
    )

    model = train_lgbm(X_train_lgbm, y_train, categorical_features)
    valid_pred = model.predict_proba(X_valid_lgbm)[:, 1]
    test_pred = model.predict_proba(X_test_lgbm)[:, 1]

    threshold_tuning = make_threshold_tuning_table(task, "LightGBM", y_valid, valid_pred)
    threshold = select_best_threshold(threshold_tuning)
    threshold_tuning["selected"] = threshold_tuning["threshold"].eq(threshold)

    model_path = MODELS_DIR / f"{safe_task_name(task)}.joblib"
    joblib.dump(model, model_path)

    metrics = make_metric_rows(task, "LightGBM", model_path, y_valid, y_test, valid_pred, test_pred, threshold)
    predictions = make_prediction_sample(task, "LightGBM", y_test, test_pred, threshold)
    output = {
        "model": model,
        "X_test": X_test_lgbm,
        "y_test": y_test,
        "test_pred": test_pred,
    }
    return output, metrics, threshold_tuning, predictions


def make_metric_rows(
    task: str,
    model_name: str,
    model_path: Path,
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """valid/test 지표를 같은 포맷의 DataFrame으로 만든다."""
    rows = []
    for split, y_true, pred in [("valid", y_valid, valid_pred), ("test", y_test, test_pred)]:
        row = {"task": task, "model": model_name, "split": split}
        row.update(evaluate_binary_classifier(y_true, pred, threshold))
        row["model_file"] = str(model_path)
        rows.append(row)
    return pd.DataFrame(rows)


def make_prediction_sample(task: str, model_name: str, y_test: pd.Series, test_pred: np.ndarray, threshold: float) -> pd.DataFrame:
    """검토용 test prediction sample을 만든다."""
    return pd.DataFrame(
        {
            "task": task,
            "model": model_name,
            "split": "test",
            "y_true": y_test.to_numpy(),
            "pred_proba": test_pred,
            "pred_label": (test_pred >= threshold).astype(int),
        }
    ).head(PREDICTION_SAMPLE_SIZE)


def run_model_comparison_for_task(
    df: pd.DataFrame,
    target: str,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """한 task에 대해 RF/LGBM/XGB를 학습하고 비교표를 만든다."""
    features, categorical_features, numeric_features = get_feature_columns()
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
    ) = split_by_time(df, target, features)

    split_summary = make_split_summary(train_df, valid_df, test_df, target)
    split_summary.insert(0, "task", task)

    metric_tables = []
    threshold_tables = []
    prediction_tables = []

    print(f"   Train {task} - LightGBM")
    lgbm_output, lgbm_metrics, lgbm_thresholds, lgbm_predictions = evaluate_lgbm_candidate(
        task=task,
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_features=categorical_features,
    )
    metric_tables.append(lgbm_metrics)
    threshold_tables.append(lgbm_thresholds)
    prediction_tables.append(lgbm_predictions)

    for model_name, estimator in build_model_candidates(y_train).items():
        print(f"   Train {task} - {model_name}")
        metrics, thresholds, predictions = evaluate_sklearn_candidate(
            task=task,
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
        threshold_tables.append(thresholds)
        prediction_tables.append(predictions)

    return (
        split_summary,
        pd.concat(metric_tables, ignore_index=True),
        pd.concat(threshold_tables, ignore_index=True),
        pd.concat(prediction_tables, ignore_index=True),
        lgbm_output,
    )


def sample_for_analysis(X: pd.DataFrame, y: pd.Series, max_size: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    """XAI 계산 시간을 줄이기 위해 test set에서 재현 가능한 sample을 추출한다."""
    if len(X) <= max_size:
        return X.copy(), y.copy()
    sample_idx = X.sample(n=max_size, random_state=random_state).index
    return X.loc[sample_idx].copy(), y.loc[sample_idx].copy()


def permutation_importance_by_group(
    model: LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: dict[str, list[str]],
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """feature group별로 값을 섞었을 때 PR-AUC가 얼마나 떨어지는지 계산한다."""
    baseline_pred = model.predict_proba(X)[:, 1]
    baseline_pr_auc = average_precision_score(y, baseline_pred)

    rows = []
    rng = np.random.default_rng(random_state)

    for group_name, group_features in feature_groups.items():
        scores = []
        for _ in range(repeats):
            X_perm = X.copy()
            for col in group_features:
                shuffled = rng.permutation(X_perm[col].to_numpy())
                if isinstance(X_perm[col].dtype, pd.CategoricalDtype):
                    X_perm[col] = pd.Categorical(shuffled, categories=X_perm[col].cat.categories)
                else:
                    X_perm[col] = shuffled

            perm_pred = model.predict_proba(X_perm)[:, 1]
            scores.append(average_precision_score(y, perm_pred))

        scores_arr = np.asarray(scores)
        rows.append(
            {
                "group": group_name,
                "n_features": len(group_features),
                "baseline_pr_auc": float(baseline_pr_auc),
                "permuted_pr_auc_mean": float(scores_arr.mean()),
                "importance_mean": float(baseline_pr_auc - scores_arr.mean()),
                "importance_std": float(scores_arr.std(ddof=0)),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def get_binary_shap_values(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """LightGBM binary classifier의 positive class SHAP 값을 반환한다."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    return np.asarray(shap_values)


def summarize_shap_by_feature(X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """feature별 SHAP 크기와 방향을 요약한다."""
    rows = []
    for idx, feature in enumerate(X.columns):
        values = shap_values[:, idx]
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(values))),
                "mean_shap": float(np.mean(values)),
                "std_shap": float(np.std(values)),
                "min_shap": float(np.min(values)),
                "max_shap": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def summarize_shap_by_group(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """feature group별 SHAP 절댓값 합계를 요약한다."""
    feature_to_idx = {feature: idx for idx, feature in enumerate(X.columns)}
    rows = []

    for group_name, group_features in feature_groups.items():
        idx = [feature_to_idx[feature] for feature in group_features if feature in feature_to_idx]
        group_shap = shap_values[:, idx].sum(axis=1)
        group_abs = np.abs(shap_values[:, idx]).sum(axis=1)
        rows.append(
            {
                "group": group_name,
                "n_features": len(idx),
                "mean_abs_shap_sum": float(group_abs.mean()),
                "mean_shap_sum": float(group_shap.mean()),
                "median_shap_sum": float(np.median(group_shap)),
                "min_shap_sum": float(group_shap.min()),
                "max_shap_sum": float(group_shap.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap_sum", ascending=False).reset_index(drop=True)


def summarize_categorical_shap(
    X: pd.DataFrame,
    y: pd.Series,
    pred_proba: np.ndarray,
    shap_values: np.ndarray,
    categorical_features: list[str],
) -> pd.DataFrame:
    """범주형 feature의 category별 평균 SHAP 값을 계산한다."""
    feature_to_idx = {feature: idx for idx, feature in enumerate(X.columns)}
    rows = []

    for feature in categorical_features:
        idx = feature_to_idx[feature]
        tmp = pd.DataFrame(
            {
                "feature": feature,
                "category": X[feature].astype(str).to_numpy(),
                "target": y.to_numpy(),
                "pred_proba": pred_proba,
                "shap": shap_values[:, idx],
                "abs_shap": np.abs(shap_values[:, idx]),
            }
        )
        grouped = (
            tmp.groupby(["feature", "category"], observed=False)
            .agg(
                n=("target", "size"),
                target_rate=("target", "mean"),
                mean_pred_proba=("pred_proba", "mean"),
                mean_shap=("shap", "mean"),
                mean_abs_shap=("abs_shap", "mean"),
            )
            .reset_index()
        )
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def summarize_recency_bins(
    X: pd.DataFrame,
    y: pd.Series,
    pred_proba: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """recency feature를 구간화해서 SHAP 방향을 요약한다."""
    recency_features = ["days_since_last_email", "days_since_last_open", "days_since_last_click"]
    flag_map = {
        "days_since_last_email": "has_prior_email",
        "days_since_last_open": "has_prior_open",
        "days_since_last_click": "has_prior_click",
    }
    feature_to_idx = {feature: idx for idx, feature in enumerate(X.columns)}
    rows = []

    for feature in recency_features:
        idx = feature_to_idx[feature]
        flag = flag_map[feature]
        tmp = pd.DataFrame(
            {
                "value": X[feature].to_numpy(),
                "has_prior": X[flag].to_numpy(),
                "target": y.to_numpy(),
                "pred_proba": pred_proba,
                "shap": shap_values[:, idx],
                "abs_shap": np.abs(shap_values[:, idx]),
            }
        )
        tmp = tmp[tmp["has_prior"].eq(1)].copy()
        tmp["bin"] = pd.cut(
            tmp["value"],
            bins=[-0.001, 1, 3, 7, np.inf],
            labels=["0-1d", "2-3d", "4-7d", "7d+"],
            include_lowest=True,
        )
        grouped = (
            tmp.groupby("bin", observed=False)
            .agg(
                n=("target", "size"),
                target_rate=("target", "mean"),
                mean_pred_proba=("pred_proba", "mean"),
                mean_value=("value", "mean"),
                mean_shap=("shap", "mean"),
                mean_abs_shap=("abs_shap", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "feature", feature)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def summarize_frequency_bins(
    X: pd.DataFrame,
    y: pd.Series,
    pred_proba: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """7일 빈도 feature를 구간화해서 SHAP 방향을 요약한다."""
    frequency_features = ["email_count_7d", "open_count_7d", "click_count_7d"]
    feature_to_idx = {feature: idx for idx, feature in enumerate(X.columns)}
    rows = []

    for feature in frequency_features:
        idx = feature_to_idx[feature]
        tmp = pd.DataFrame(
            {
                "value": X[feature].to_numpy(),
                "target": y.to_numpy(),
                "pred_proba": pred_proba,
                "shap": shap_values[:, idx],
                "abs_shap": np.abs(shap_values[:, idx]),
            }
        )
        tmp["bin"] = pd.cut(
            tmp["value"],
            bins=[-0.001, 0, 1, 2, 3, 5, np.inf],
            labels=["0", "1", "2", "3", "4-5", "6+"],
            include_lowest=True,
        )
        grouped = (
            tmp.groupby("bin", observed=False)
            .agg(
                n=("target", "size"),
                target_rate=("target", "mean"),
                mean_pred_proba=("pred_proba", "mean"),
                mean_value=("value", "mean"),
                mean_shap=("shap", "mean"),
                mean_abs_shap=("abs_shap", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "feature", feature)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def summarize_prior_flags(
    X: pd.DataFrame,
    y: pd.Series,
    pred_proba: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """prior flag feature의 0/1 값별 SHAP 방향을 요약한다."""
    prior_features = ["has_prior_email", "has_prior_open", "has_prior_click"]
    feature_to_idx = {feature: idx for idx, feature in enumerate(X.columns)}
    rows = []

    for feature in prior_features:
        idx = feature_to_idx[feature]
        tmp = pd.DataFrame(
            {
                "value": X[feature].to_numpy(),
                "target": y.to_numpy(),
                "pred_proba": pred_proba,
                "shap": shap_values[:, idx],
                "abs_shap": np.abs(shap_values[:, idx]),
            }
        )
        grouped = (
            tmp.groupby("value", observed=False)
            .agg(
                n=("target", "size"),
                target_rate=("target", "mean"),
                mean_pred_proba=("pred_proba", "mean"),
                mean_shap=("shap", "mean"),
                mean_abs_shap=("abs_shap", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "feature", feature)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def make_shap_bar_plot(shap_summary: pd.DataFrame, model_name: str, filename: str) -> Path:
    """상위 SHAP feature importance plot을 저장한다."""
    plot_df = shap_summary.sort_values("mean_abs_shap", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#4C78A8")
    ax.set_title(f"{model_name} SHAP Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("mean(|SHAP|)")
    ax.grid(axis="x", alpha=0.25)
    return savefig(filename)


def make_permutation_plot(perm_df: pd.DataFrame, model_name: str, filename: str) -> Path:
    """feature group permutation importance plot을 저장한다."""
    plot_df = perm_df.sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(plot_df["group"], plot_df["importance_mean"], color="#F58518")
    ax.set_title(f"{model_name} Permutation Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("PR-AUC decrease")
    ax.grid(axis="x", alpha=0.25)
    return savefig(filename)


def savefig(name: str) -> Path:
    """현재 matplotlib figure를 metrics 폴더에 저장한다."""
    path = METRICS_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def run_xai_pipeline(
    output: dict[str, object],
    model_name: str,
    categorical_features: list[str],
    feature_groups: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """LightGBM 모델 하나에 대해 permutation importance와 SHAP 요약을 생성한다."""
    model = output["model"]
    X_test = output["X_test"]
    y_test = output["y_test"]

    X_perm, y_perm = sample_for_analysis(X_test, y_test, PERMUTATION_SAMPLE_SIZE, RANDOM_STATE)
    perm_df = permutation_importance_by_group(
        model=model,
        X=X_perm,
        y=y_perm,
        feature_groups=feature_groups,
        repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
    )

    X_shap, y_shap = sample_for_analysis(X_test, y_test, SHAP_SAMPLE_SIZE, RANDOM_STATE)
    shap_pred = model.predict_proba(X_shap)[:, 1]
    shap_values = get_binary_shap_values(model, X_shap)

    shap_feature_df = summarize_shap_by_feature(X_shap, shap_values)
    shap_group_df = summarize_shap_by_group(X_shap, shap_values, feature_groups)
    categorical_shap_df = summarize_categorical_shap(X_shap, y_shap, shap_pred, shap_values, categorical_features)
    recency_shap_df = summarize_recency_bins(X_shap, y_shap, shap_pred, shap_values)
    frequency_shap_df = summarize_frequency_bins(X_shap, y_shap, shap_pred, shap_values)
    prior_shap_df = summarize_prior_flags(X_shap, y_shap, shap_pred, shap_values)

    prefix = model_name.lower().replace("-", "_").replace(" ", "_")
    make_permutation_plot(perm_df, model_name, f"{prefix}_permutation_importance.png")
    make_shap_bar_plot(shap_feature_df, model_name, f"{prefix}_shap_feature_importance.png")

    return {
        "permutation": perm_df,
        "shap_feature": shap_feature_df,
        "shap_group": shap_group_df,
        "categorical_shap": categorical_shap_df,
        "recency_shap": recency_shap_df,
        "frequency_shap": frequency_shap_df,
        "prior_shap": prior_shap_df,
    }


def write_outputs(
    split_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    thresholds: pd.DataFrame,
    predictions: pd.DataFrame,
    open_xai: dict[str, pd.DataFrame],
    click_xai: dict[str, pd.DataFrame],
) -> None:
    """모델 비교 결과와 LightGBM XAI 결과를 파일로 저장한다."""
    comparison_cols = [
        "task",
        "model",
        "split",
        "threshold",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_positive_rate",
        "pred_positive_rate",
        "model_file",
    ]
    threshold_cols = [
        "task",
        "model",
        "split",
        "threshold",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "pred_positive_rate",
        "base_positive_rate",
        "selected",
    ]

    comparison = comparison[comparison_cols].sort_values(["task", "model", "split"]).reset_index(drop=True)
    thresholds = thresholds[threshold_cols].sort_values(["task", "model", "threshold"]).reset_index(drop=True)
    best_models = (
        comparison[comparison["split"].eq("test")]
        .sort_values(["task", "pr_auc", "roc_auc", "f1"], ascending=[True, False, False, False])
        .groupby("task", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    split_summary.to_csv(METRICS_DIR / "model_01_split_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(METRICS_DIR / "model_02_model_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(METRICS_DIR / "model_02_metrics.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(METRICS_DIR / "model_03_threshold_tuning.csv", index=False, encoding="utf-8-sig")
    best_models.to_csv(METRICS_DIR / "model_04_best_models.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(METRICS_DIR / "model_06_prediction_sample.csv", index=False, encoding="utf-8-sig")

    for model_key, xai in [("open", open_xai), ("click_after_open", click_xai)]:
        xai["permutation"].to_csv(METRICS_DIR / f"xai_{model_key}_01_permutation.csv", index=False, encoding="utf-8-sig")
        xai["shap_feature"].to_csv(METRICS_DIR / f"xai_{model_key}_02_shap_feature.csv", index=False, encoding="utf-8-sig")
        xai["shap_group"].to_csv(METRICS_DIR / f"xai_{model_key}_03_shap_group.csv", index=False, encoding="utf-8-sig")
        xai["categorical_shap"].to_csv(METRICS_DIR / f"xai_{model_key}_04_categorical_shap.csv", index=False, encoding="utf-8-sig")
        xai["recency_shap"].to_csv(METRICS_DIR / f"xai_{model_key}_05_recency_shap.csv", index=False, encoding="utf-8-sig")
        xai["frequency_shap"].to_csv(METRICS_DIR / f"xai_{model_key}_06_frequency_shap.csv", index=False, encoding="utf-8-sig")
        xai["prior_shap"].to_csv(METRICS_DIR / f"xai_{model_key}_07_prior_shap.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "model_comparison": "RandomForest, LightGBM, XGBoost",
        "xai_scope": "LightGBM only",
        "best_models_by_test_pr_auc": best_models.to_dict(orient="records"),
        "metrics_dir": str(METRICS_DIR),
        "models_dir": str(MODELS_DIR),
    }
    (METRICS_DIR / "model_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def main() -> None:
    """모델 비교와 LightGBM-only XAI 전체 pipeline을 실행한다."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    ensure_output_dirs()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:.6f}".format)

    print("1/5 Load processed dataset")
    df = pd.read_pickle(DATA_PATH)
    features, categorical_features, _ = get_feature_columns()
    feature_groups = get_feature_groups()
    print(f"   rows: {len(df):,}")
    print(f"   model_features: {len(features)}")
    print(f"   period: {df['sent_at'].min()} ~ {df['sent_at'].max()}")

    print("\n2/5 Train and evaluate Open models")
    open_split, open_metrics, open_thresholds, open_predictions, open_lgbm = run_model_comparison_for_task(
        df=df,
        target="target_opened",
        task="Open",
    )

    print("\n3/5 Train and evaluate Click-after-Open models")
    click_df = df[df["target_opened"].eq(1)].copy()
    click_split, click_metrics, click_thresholds, click_predictions, click_lgbm = run_model_comparison_for_task(
        df=click_df,
        target="target_clicked",
        task="Click-after-Open",
    )

    print("\n4/5 Run LightGBM-only XAI")
    open_xai = run_xai_pipeline(open_lgbm, "Open", categorical_features, feature_groups)
    click_xai = run_xai_pipeline(click_lgbm, "Click-after-Open", categorical_features, feature_groups)

    print("\n5/5 Save modeling and XAI outputs")
    split_summary = pd.concat([open_split, click_split], ignore_index=True)
    comparison = pd.concat([open_metrics, click_metrics], ignore_index=True)
    thresholds = pd.concat([open_thresholds, click_thresholds], ignore_index=True)
    predictions = pd.concat([open_predictions, click_predictions], ignore_index=True)
    write_outputs(split_summary, comparison, thresholds, predictions, open_xai, click_xai)

    print("\n[Test model comparison]")
    print(comparison[comparison["split"].eq("test")].sort_values(["task", "model"]).to_string(index=False))
    print(f"\nSaved models: {MODELS_DIR.resolve()}")
    print(f"Saved metrics: {METRICS_DIR.resolve()}")


if __name__ == "__main__":
    main()
