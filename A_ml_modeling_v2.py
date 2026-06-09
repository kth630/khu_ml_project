from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def resolve_project_root(required_files: list[str]) -> Path:
    # 스크립트 파일이 있는 폴더를 후보 경로에 추가한다.
    script_dir = Path(__file__).resolve().parent

    # 터미널의 현재 작업 폴더를 후보 경로에 추가한다.
    cwd = Path.cwd().resolve()

    # VS Code가 임시 폴더에서 실행할 때를 대비해 실제 프로젝트 경로 후보를 추가한다.
    known_project_dir = (
        Path.home()
        / "Desktop"
        / "학교"
        / "26-1"
        / "머신러닝기초및응용"
        / "머신러닝_프젝"
        / "ecommerce"
    )

    # 각 후보 경로와 그 상위 폴더들을 순서대로 검사한다.
    candidates = [script_dir, cwd, known_project_dir]
    candidates.extend(script_dir.parents)
    candidates.extend(cwd.parents)

    # 필요한 파일이 모두 존재하는 첫 번째 폴더를 프로젝트 루트로 사용한다.
    for candidate in candidates:
        if all((candidate / name).exists() for name in required_files):
            return candidate

    # 어떤 후보에서도 찾지 못하면 확인한 경로 목록과 함께 에러를 발생시킨다.
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Required project files were not found.\n"
        f"Required files: {required_files}\n"
        f"Checked paths:\n{checked}"
    )


ROOT = resolve_project_root(["A_ml_dataset.parquet"])
DATA_PATH = ROOT / "A_ml_dataset.parquet"
OUTPUT_DIR = ROOT / "A_ml_modeling_outputs"
RANDOM_STATE = 42
PERMUTATION_REPEATS = 3
PERMUTATION_SAMPLE_SIZE = 100_000
SHAP_SAMPLE_SIZE = 100_000
THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)


# matplotlib에서 한글과 음수 기호가 깨지지 않도록 기본 폰트 설정을 지정한다.
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir() -> None:
    # 모델링 결과 표와 그림을 저장할 폴더를 만든다.
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 이전 실행에서 생성된 파일이 섞이지 않도록 출력 폴더 안의 기존 파일을 삭제한다.
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def get_feature_columns() -> tuple[list[str], list[str], list[str]]:
    # 모델에 사용할 범주형 변수 목록을 정의한다.
    categorical_features = [
        "topic",
        "provider_group",
        "send_hour",
        "send_dow",
    ]

    # 모델에 사용할 수치형 변수 목록을 정의한다.
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

    # 범주형 변수와 수치형 변수를 합쳐 전체 feature 목록을 만든다.
    features = categorical_features + numeric_features
    return features, categorical_features, numeric_features


def get_feature_groups() -> dict[str, list[str]]:
    # permutation importance와 group SHAP 계산에 사용할 변수 그룹을 정의한다.
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


def save_table(df: pd.DataFrame, name: str) -> Path:
    # DataFrame을 CSV 파일로 저장한다.
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def savefig(name: str) -> Path:
    # 현재 matplotlib figure를 PNG 파일로 저장하고 figure를 닫는다.
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def prepare_categories(frames: list[pd.DataFrame], categorical_features: list[str]) -> list[pd.DataFrame]:
    # train/valid/test 전체에서 등장한 범주 level을 컬럼별로 수집한다.
    category_levels = {}
    for col in categorical_features:
        levels = []
        for frame in frames:
            levels.extend(frame[col].dropna().astype(str).unique().tolist())
        category_levels[col] = sorted(set(levels))

    # 각 DataFrame의 범주형 컬럼을 동일한 categories를 갖는 pandas Categorical로 변환한다.
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
    # 전체 데이터를 발송 시각 기준으로 정렬한다.
    ordered = df.sort_values("sent_at").reset_index(drop=True)
    n = len(ordered)

    # 시간 순서 기준으로 60% train, 20% validation, 20% test 경계를 계산한다.
    train_end = int(n * 0.6)
    valid_end = int(n * 0.8)

    # 정렬된 데이터를 train/validation/test로 자른다.
    train_df = ordered.iloc[:train_end].copy()
    valid_df = ordered.iloc[train_end:valid_end].copy()
    test_df = ordered.iloc[valid_end:].copy()

    # feature matrix와 target vector를 분리한다.
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df[target].astype(int).copy()
    y_valid = valid_df[target].astype(int).copy()
    y_test = test_df[target].astype(int).copy()

    return X_train, X_valid, X_test, y_train, y_valid, y_test, train_df, valid_df, test_df


def sample_for_analysis(X: pd.DataFrame, y: pd.Series, max_size: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    # max_size보다 작으면 전체 데이터를 그대로 사용한다.
    if len(X) <= max_size:
        return X.copy(), y.copy()

    # max_size보다 크면 재현 가능한 random sample을 추출한다.
    sample_idx = X.sample(n=max_size, random_state=random_state).index
    return X.loc[sample_idx].copy(), y.loc[sample_idx].copy()


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, categorical_features: list[str]) -> LGBMClassifier:
    # balanced LightGBM binary classifier를 생성한다.
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

    # 범주형 feature 이름을 LightGBM에 전달하여 모델을 학습한다.
    model.fit(X_train, y_train, categorical_feature=categorical_features)
    return model


def evaluate_binary_classifier(y_true: pd.Series, pred_proba: np.ndarray, threshold: float) -> dict[str, float]:
    # 지정한 threshold 기준으로 예측 class를 만든다.
    y_pred = (np.asarray(pred_proba) >= threshold).astype(int)

    # 기본 이진 분류 지표를 계산한다.
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, pred_proba),
        "pr_auc": average_precision_score(y_true, pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_positive_rate": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(y_pred)),
    }


def make_threshold_tuning_table(model_name: str, y_true: pd.Series, pred_proba: np.ndarray) -> pd.DataFrame:
    # Series/array를 numpy array로 변환한다.
    y_true_arr = np.asarray(y_true).astype(int)
    pred_proba_arr = np.asarray(pred_proba)
    base_positive_rate = float(np.mean(y_true_arr))

    # validation set에서 threshold 후보별 분류 성능을 계산한다.
    rows = []
    for threshold in THRESHOLD_GRID:
        y_pred = (pred_proba_arr >= threshold).astype(int)
        precision = precision_score(y_true_arr, y_pred, zero_division=0)
        rows.append(
            {
                "model": model_name,
                "split": "valid",
                "threshold": threshold,
                "precision": precision,
                "precision_improvement_pp": precision - base_positive_rate,
                "precision_lift": precision / base_positive_rate if base_positive_rate else np.nan,
                "recall": recall_score(y_true_arr, y_pred, zero_division=0),
                "f1": f1_score(y_true_arr, y_pred, zero_division=0),
                "pred_positive_rate": float(np.mean(y_pred)),
                "base_positive_rate": base_positive_rate,
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(threshold_tuning: pd.DataFrame) -> float:
    # F1이 가장 높은 후보를 우선하고, 동률이면 precision과 threshold가 높은 후보를 선택한다.
    selected = threshold_tuning.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(selected["threshold"])


def make_split_summary(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> pd.DataFrame:
    # split별 행 수, 기간, positive rate를 표로 정리한다.
    rows = []
    for split, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        rows.append(
            {
                "split": split,
                "rows": len(part),
                "start_at": part["sent_at"].min(),
                "end_at": part["sent_at"].max(),
                "positive_rate": part[target].mean(),
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_by_group(
    model: LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: dict[str, list[str]],
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    # 원본 데이터 기준 PR-AUC를 계산한다.
    baseline_pred = model.predict_proba(X)[:, 1]
    baseline_pr_auc = average_precision_score(y, baseline_pred)

    # 그룹별 permutation importance 결과를 저장할 리스트를 만든다.
    rows = []
    rng = np.random.default_rng(random_state)

    for group_name, group_features in feature_groups.items():
        scores = []

        # 같은 그룹을 repeats번 섞어서 PR-AUC 하락폭의 평균과 표준편차를 계산한다.
        for _ in range(repeats):
            X_perm = X.copy()
            for col in group_features:
                # 컬럼 값을 무작위로 섞는다.
                shuffled = rng.permutation(X_perm[col].to_numpy())

                # categorical 컬럼은 섞은 뒤에도 원래 categories 정보를 유지한다.
                if isinstance(X_perm[col].dtype, pd.CategoricalDtype):
                    X_perm[col] = pd.Categorical(shuffled, categories=X_perm[col].cat.categories)
                else:
                    X_perm[col] = shuffled

            perm_pred = model.predict_proba(X_perm)[:, 1]
            permuted_pr_auc = average_precision_score(y, perm_pred)
            scores.append(permuted_pr_auc)

        scores = np.asarray(scores)
        rows.append(
            {
                "group": group_name,
                "n_features": len(group_features),
                "baseline_pr_auc": baseline_pr_auc,
                "permuted_pr_auc_mean": scores.mean(),
                "importance_mean": baseline_pr_auc - scores.mean(),
                "importance_std": scores.std(ddof=0),
            }
        )

    # importance가 큰 그룹부터 정렬한다.
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def get_binary_shap_values(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    # LightGBM tree model용 SHAP explainer를 생성한다.
    explainer = shap.TreeExplainer(model)

    # SHAP 값을 계산한다.
    shap_values = explainer.shap_values(X)

    # shap 버전에 따라 binary classification 결과가 list로 나올 수 있어 positive class만 선택한다.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Explanation 객체가 반환되는 경우 values 배열만 꺼낸다.
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    return np.asarray(shap_values)


def summarize_shap_by_feature(X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    # feature별 SHAP 평균, 절댓값 평균, 표준편차, 최소/최대값을 계산한다.
    rows = []
    for i, feature in enumerate(X.columns):
        vals = shap_values[:, i]
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": np.mean(np.abs(vals)),
                "mean_shap": np.mean(vals),
                "std_shap": np.std(vals),
                "min_shap": np.min(vals),
                "max_shap": np.max(vals),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def summarize_shap_by_group(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    # feature 이름에서 SHAP 배열 column index를 찾기 위한 mapping을 만든다.
    feature_to_idx = {feature: i for i, feature in enumerate(X.columns)}

    # 그룹별 SHAP 절댓값 합과 SHAP 합의 통계량을 계산한다.
    rows = []
    for group_name, group_features in feature_groups.items():
        idx = [feature_to_idx[feature] for feature in group_features if feature in feature_to_idx]
        group_shap = shap_values[:, idx].sum(axis=1)
        group_abs = np.abs(shap_values[:, idx]).sum(axis=1)
        rows.append(
            {
                "group": group_name,
                "n_features": len(idx),
                "mean_abs_shap_sum": group_abs.mean(),
                "mean_shap_sum": group_shap.mean(),
                "median_shap_sum": np.median(group_shap),
                "min_shap_sum": group_shap.min(),
                "max_shap_sum": group_shap.max(),
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
    # feature 이름에서 SHAP 배열 column index를 찾기 위한 mapping을 만든다.
    feature_to_idx = {feature: i for i, feature in enumerate(X.columns)}

    # 범주형 변수의 category별 target rate, 평균 예측확률, 평균 SHAP을 계산한다.
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
    # recency 변수와 해당 prior flag를 매핑한다.
    recency_features = ["days_since_last_email", "days_since_last_open", "days_since_last_click"]
    flag_map = {
        "days_since_last_email": "has_prior_email",
        "days_since_last_open": "has_prior_open",
        "days_since_last_click": "has_prior_click",
    }
    feature_to_idx = {feature: i for i, feature in enumerate(X.columns)}

    # 과거 행동이 있는 행에 대해서만 recency 구간별 SHAP을 계산한다.
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
    # frequency 변수 목록을 정의한다.
    frequency_features = ["email_count_7d", "open_count_7d", "click_count_7d"]
    feature_to_idx = {feature: i for i, feature in enumerate(X.columns)}

    # count 구간별 target rate, 평균 예측확률, 평균 SHAP을 계산한다.
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
    # prior flag 변수 목록을 정의한다.
    prior_features = ["has_prior_email", "has_prior_open", "has_prior_click"]
    feature_to_idx = {feature: i for i, feature in enumerate(X.columns)}

    # flag 값 0/1별 target rate, 평균 예측확률, 평균 SHAP을 계산한다.
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
    # SHAP 절댓값 평균 기준 상위 feature를 선택한다.
    plot_df = shap_summary.sort_values("mean_abs_shap", ascending=True).tail(15)

    # feature별 mean_abs_shap을 가로 막대그래프로 그린다.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#4C78A8")
    ax.set_title(f"{model_name} SHAP Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("mean(|SHAP|)")
    ax.grid(axis="x", alpha=0.25)
    return savefig(filename)


def make_permutation_plot(perm_df: pd.DataFrame, model_name: str, filename: str) -> Path:
    # permutation importance가 낮은 그룹부터 높은 그룹 순서로 정렬한다.
    plot_df = perm_df.sort_values("importance_mean", ascending=True)

    # 변수 그룹별 PR-AUC 하락폭을 가로 막대그래프로 그린다.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(plot_df["group"], plot_df["importance_mean"], color="#F58518")
    ax.set_title(f"{model_name} Permutation Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("PR-AUC decrease")
    ax.grid(axis="x", alpha=0.25)
    return savefig(filename)


def run_model_pipeline(df: pd.DataFrame, target: str, model_name: str) -> dict[str, object]:
    # 모델 feature 목록과 범주형 feature 목록을 가져온다.
    features, categorical_features, _ = get_feature_columns()

    # 시간 기준 train/validation/test split을 수행한다.
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

    # 범주형 feature를 pandas Categorical 타입으로 변환한다.
    X_train, X_valid, X_test = prepare_categories(
        [X_train, X_valid, X_test],
        categorical_features,
    )

    # LightGBM 모델을 학습한다.
    model = train_lgbm(X_train, y_train, categorical_features)

    # validation/test 예측 확률을 계산한다.
    valid_pred = model.predict_proba(X_valid)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    # validation set에서 threshold 후보별 분류 성능을 계산하고 F1 기준 최적 threshold를 선택한다.
    threshold_tuning = make_threshold_tuning_table(model_name, y_valid, valid_pred)
    threshold = select_best_threshold(threshold_tuning)
    threshold_tuning["selected"] = threshold_tuning["threshold"].eq(threshold)

    # validation/test 기본 성능 지표를 계산한다.
    metric_rows = []
    for split, y_true, pred in [("valid", y_valid, valid_pred), ("test", y_test, test_pred)]:
        row = {"model": model_name, "split": split}
        row.update(evaluate_binary_classifier(y_true, pred, threshold))
        metric_rows.append(row)

    # split별 기간과 target rate를 정리한다.
    split_summary = make_split_summary(train_df, valid_df, test_df, target)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "test_pred": test_pred,
        "metrics": pd.DataFrame(metric_rows),
        "threshold_tuning": threshold_tuning,
        "split_summary": split_summary,
    }


def run_xai_pipeline(
    output: dict[str, object],
    model_name: str,
    categorical_features: list[str],
    feature_groups: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    # test set과 예측 결과를 꺼낸다.
    model = output["model"]
    X_test = output["X_test"]
    y_test = output["y_test"]

    # permutation importance 계산용 test sample을 만든다.
    X_perm, y_perm = sample_for_analysis(
        X_test,
        y_test,
        max_size=PERMUTATION_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    # 변수 그룹별 permutation importance를 계산한다.
    perm_df = permutation_importance_by_group(
        model=model,
        X=X_perm,
        y=y_perm,
        feature_groups=feature_groups,
        repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
    )

    # SHAP 계산용 test sample을 만든다.
    X_shap, y_shap = sample_for_analysis(
        X_test,
        y_test,
        max_size=SHAP_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    # SHAP sample에 대한 예측 확률과 SHAP 값을 계산한다.
    shap_pred = model.predict_proba(X_shap)[:, 1]
    shap_values = get_binary_shap_values(model, X_shap)

    # SHAP 결과를 feature, group, 범주, 구간 단위로 요약한다.
    shap_feature_df = summarize_shap_by_feature(X_shap, shap_values)
    shap_group_df = summarize_shap_by_group(X_shap, shap_values, feature_groups)
    categorical_shap_df = summarize_categorical_shap(X_shap, y_shap, shap_pred, shap_values, categorical_features)
    recency_shap_df = summarize_recency_bins(X_shap, y_shap, shap_pred, shap_values)
    frequency_shap_df = summarize_frequency_bins(X_shap, y_shap, shap_pred, shap_values)
    prior_shap_df = summarize_prior_flags(X_shap, y_shap, shap_pred, shap_values)

    # permutation importance와 SHAP feature importance plot을 저장한다.
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


def main() -> None:
    # 콘솔 출력에서 컬럼 생략이 덜 생기도록 pandas 출력 옵션을 설정한다.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:.6f}".format)

    # 출력 폴더를 생성한다.
    ensure_output_dir()

    # 최종 전처리 데이터셋을 로드한다.
    print("1/6 Load final dataset")
    df = pd.read_parquet(DATA_PATH)

    # feature 목록, categorical feature 목록, feature group 정보를 가져온다.
    features, categorical_features, _ = get_feature_columns()
    feature_groups = get_feature_groups()

    # 최종 데이터셋의 기본 정보를 출력한다.
    print(f"   rows: {len(df):,}")
    print(f"   columns: {len(df.columns):,}")
    print(f"   model_features: {len(features)}")
    print(f"   categorical_features: {categorical_features}")
    print(f"   period: {df['sent_at'].min()} ~ {df['sent_at'].max()}")
    print(f"   open_rate: {df['target_opened'].mean():.6f}")
    print(f"   ctr: {df['target_clicked'].mean():.6f}")
    print(f"   ctor: {df.loc[df['target_opened'].eq(1), 'target_clicked'].mean():.6f}")

    # 전체 발송 건을 대상으로 Open 모델을 학습한다.
    print("\n2/6 Train Open model")
    open_output = run_model_pipeline(df, "target_opened", "Open")

    # 오픈한 고객만 대상으로 Click-after-Open 모델을 학습한다.
    print("3/6 Train Click-after-Open model")
    click_df = df[df["target_opened"].eq(1)].copy()
    click_output = run_model_pipeline(click_df, "target_clicked", "Click-after-Open")

    # split 요약, 최종 성능, threshold 후보별 성능을 하나의 표로 합친다.
    split_summary = pd.concat(
        [
            open_output["split_summary"].assign(model="Open"),
            click_output["split_summary"].assign(model="Click-after-Open"),
        ],
        ignore_index=True,
    )
    metrics = pd.concat([open_output["metrics"], click_output["metrics"]], ignore_index=True)
    threshold_tuning = pd.concat(
        [open_output["threshold_tuning"], click_output["threshold_tuning"]],
        ignore_index=True,
    )

    # 모델링 성능 관련 표를 CSV로 저장한다.
    save_table(split_summary, "model_01_split_summary.csv")
    save_table(metrics, "model_02_metrics.csv")
    save_table(threshold_tuning, "model_03_threshold_tuning.csv")

    # 콘솔에 validation/test 최종 성능과 threshold 후보별 validation 성능을 출력한다.
    print("\n4/6 Save and print model performance")
    print("\n[Validation/Test metrics]")
    print(metrics.to_string(index=False))
    print("\n[Validation threshold tuning]")
    print(threshold_tuning.to_string(index=False))

    # Open 모델의 permutation importance와 SHAP 분석을 수행한다.
    print("\n5/6 Run Open XAI")
    open_xai = run_xai_pipeline(open_output, "Open", categorical_features, feature_groups)

    # Click-after-Open 모델의 permutation importance와 SHAP 분석을 수행한다.
    print("6/6 Run Click-after-Open XAI")
    click_xai = run_xai_pipeline(click_output, "Click-after-Open", categorical_features, feature_groups)

    # XAI 결과 표를 CSV로 저장한다.
    for model_key, xai in [("open", open_xai), ("click_after_open", click_xai)]:
        save_table(xai["permutation"], f"xai_{model_key}_01_permutation.csv")
        save_table(xai["shap_feature"], f"xai_{model_key}_02_shap_feature.csv")
        save_table(xai["shap_group"], f"xai_{model_key}_03_shap_group.csv")
        save_table(xai["categorical_shap"], f"xai_{model_key}_04_categorical_shap.csv")
        save_table(xai["recency_shap"], f"xai_{model_key}_05_recency_shap.csv")
        save_table(xai["frequency_shap"], f"xai_{model_key}_06_frequency_shap.csv")
        save_table(xai["prior_shap"], f"xai_{model_key}_07_prior_shap.csv")

    # 콘솔에 핵심 XAI 표를 출력한다.
    print("\n[Open permutation importance]")
    print(open_xai["permutation"].to_string(index=False))
    print("\n[Click-after-Open permutation importance]")
    print(click_xai["permutation"].to_string(index=False))
    print("\n[Open SHAP feature summary]")
    print(open_xai["shap_feature"].to_string(index=False))
    print("\n[Click-after-Open SHAP feature summary]")
    print(click_xai["shap_feature"].to_string(index=False))
    print(f"\nSaved outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
