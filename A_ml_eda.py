from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
OUTPUT_DIR = ROOT / "A_ml_eda_outputs"


# matplotlib에서 한글과 음수 기호가 깨지지 않도록 기본 폰트 설정을 지정한다.
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir() -> None:
    # EDA 결과 표와 그림을 저장할 폴더를 만든다.
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 이전 실행에서 생성된 파일이 섞이지 않도록 출력 폴더 안의 기존 파일을 삭제한다.
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def savefig(name: str) -> Path:
    # 현재 matplotlib figure를 PNG 파일로 저장하고 figure를 닫는다.
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    # DataFrame을 CSV 파일로 저장한다.
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def response_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    # 그룹별 발송 수, 오픈율, CTR 계산에 필요한 집계를 수행한다.
    grouped = (
        df.groupby(group_col, observed=False)
        .agg(
            n=("target_opened", "size"),
            open_rate=("target_opened", "mean"),
            ctr=("target_clicked", "mean"),
            opened=("target_opened", "sum"),
            clicked=("target_clicked", "sum"),
        )
        .reset_index()
    )

    # CTOR는 오픈 건수 중 클릭 건수의 비율로 계산한다.
    grouped["ctor"] = np.where(grouped["opened"] > 0, grouped["clicked"] / grouped["opened"], np.nan)
    return grouped.drop(columns=["opened", "clicked"])


def make_funnel_plot(df: pd.DataFrame) -> Path:
    # 전체 발송, 오픈, 클릭 건수를 계산한다.
    sent = len(df)
    opened = int(df["target_opened"].sum())
    clicked = int(df["target_clicked"].sum())
    stages = ["Sent", "Open", "Click"]
    counts = [sent, opened, clicked]

    # 퍼널을 가로 막대그래프로 표현한다.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    bars = ax.barh(stages[::-1], counts[::-1], color=colors[::-1])
    ax.set_title("Trigger Email Funnel", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of messages")
    ax.grid(axis="x", alpha=0.25)

    # 각 단계의 전체 발송 대비 비율을 막대 오른쪽에 표시한다.
    labels = {
        "Sent": "100.00%",
        "Open": f"{opened / sent:.2%}",
        "Click": f"{clicked / sent:.2%}",
    }
    max_count = max(counts)
    for bar, stage in zip(bars, stages[::-1]):
        ax.text(
            bar.get_width() + max_count * 0.018,
            bar.get_y() + bar.get_height() / 2,
            labels[stage],
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlim(0, max_count * 1.28)
    return savefig("A_ml_eda_01_funnel.png")


def make_topic_plot(topic_df: pd.DataFrame) -> Path:
    # CTOR가 낮은 topic부터 높은 topic 순서로 정렬한다.
    plot_df = topic_df.sort_values("ctor", ascending=True)

    # topic별 Open Rate와 CTOR를 가로 막대그래프로 비교한다.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = range(len(plot_df))
    ax.barh([i - 0.18 for i in y], plot_df["open_rate"], height=0.36, label="Open Rate", color="#4C78A8")
    ax.barh([i + 0.18 for i in y], plot_df["ctor"], height=0.36, label="CTOR", color="#F58518")
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["topic"])
    ax.set_xlabel("Response rate")
    ax.set_title("Topic Open Rate / CTOR", fontsize=16, fontweight="bold")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return savefig("A_ml_eda_02_topic_rates.png")


def make_provider_plot(provider_df: pd.DataFrame) -> Path:
    # CTOR가 낮은 provider부터 높은 provider 순서로 정렬한다.
    plot_df = provider_df.sort_values("ctor", ascending=True)

    # provider별 Open Rate와 CTOR를 가로 막대그래프로 비교한다.
    fig, ax = plt.subplots(figsize=(9.5, 5))
    y = range(len(plot_df))
    ax.barh([i - 0.18 for i in y], plot_df["open_rate"], height=0.36, label="Open Rate", color="#4C78A8")
    ax.barh([i + 0.18 for i in y], plot_df["ctor"], height=0.36, label="CTOR", color="#F58518")
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["provider_group"])
    ax.set_xlabel("Response rate")
    ax.set_title("Provider Open Rate / CTOR", fontsize=16, fontweight="bold")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return savefig("A_ml_eda_03_provider_rates.png")


def make_send_time_plot(hour_df: pd.DataFrame, dow_df: pd.DataFrame) -> Path:
    # send_hour를 숫자형으로 변환하고 시간 순서대로 정렬한다.
    hour_df = hour_df.copy()
    hour_df["send_hour_num"] = hour_df["send_hour"].astype(int)
    hour_df = hour_df.sort_values("send_hour_num")

    # 중간에 빠진 시간이 있으면 선 그래프가 끊기지 않도록 전체 시간 index를 만든다.
    full_hours = pd.DataFrame(
        {"send_hour_num": range(hour_df["send_hour_num"].min(), hour_df["send_hour_num"].max() + 1)}
    )
    hour_df = full_hours.merge(hour_df, on="send_hour_num", how="left")

    # 빠진 시간의 Open Rate와 CTOR는 주변 값 기준으로 보간한다.
    hour_df[["open_rate", "ctor"]] = hour_df[["open_rate", "ctor"]].interpolate(limit_direction="both")

    # send_dow를 숫자형으로 변환하고 요일 순서대로 정렬한다.
    dow_df = dow_df.copy()
    dow_df["send_dow_num"] = dow_df["send_dow"].astype(int)
    dow_df = dow_df.sort_values("send_dow_num")

    # 발송 시간과 발송 요일별 Open Rate/CTOR 선 그래프를 그린다.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(hour_df["send_hour_num"], hour_df["open_rate"], marker="o", label="Open Rate")
    axes[0].plot(hour_df["send_hour_num"], hour_df["ctor"], marker="o", label="CTOR")
    axes[0].set_title("Response Rate by Send Hour", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("send_hour")
    axes[0].set_ylabel("Response rate")
    axes[0].set_xticks(hour_df["send_hour_num"])
    axes[0].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].plot(dow_df["send_dow_num"], dow_df["open_rate"], marker="o", label="Open Rate")
    axes[1].plot(dow_df["send_dow_num"], dow_df["ctor"], marker="o", label="CTOR")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_labels)
    axes[1].set_title("Response Rate by Send Day", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("send_dow")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")
    return savefig("A_ml_eda_04_send_time_rates.png")


def make_recency_bins(df: pd.DataFrame) -> pd.DataFrame:
    # recency 변수와 해당 prior flag를 매핑한다.
    features = ["days_since_last_email", "days_since_last_open", "days_since_last_click"]
    flags = {
        "days_since_last_email": "has_prior_email",
        "days_since_last_open": "has_prior_open",
        "days_since_last_click": "has_prior_click",
    }

    # 각 recency 변수별 no prior 및 경과일 구간별 반응률을 계산한다.
    rows = []
    for feature in features:
        flag = flags[feature]

        # 과거 행동이 없는 행을 no prior 구간으로 집계한다.
        no_prior = df[df[flag].eq(0)]
        rows.append(
            {
                "feature": feature,
                "bin": "no prior",
                "n": len(no_prior),
                "open_rate": no_prior["target_opened"].mean(),
                "ctr": no_prior["target_clicked"].mean(),
                "ctor": no_prior.loc[no_prior["target_opened"].eq(1), "target_clicked"].mean(),
            }
        )

        # 과거 행동이 있는 행은 경과일 기준 구간으로 나눈다.
        part = df[df[flag].eq(1)].copy()
        part["bin"] = pd.cut(
            part[feature],
            bins=[-0.001, 1, 3, 7, np.inf],
            labels=["0-1d", "2-3d", "4-7d", "7d+"],
            include_lowest=True,
        )

        # 구간별 n, Open Rate, CTR, CTOR를 계산한다.
        tmp = (
            part.groupby("bin", observed=False)
            .agg(
                n=("target_opened", "size"),
                open_rate=("target_opened", "mean"),
                ctr=("target_clicked", "mean"),
                opened=("target_opened", "sum"),
                clicked=("target_clicked", "sum"),
            )
            .reset_index()
        )
        tmp["ctor"] = np.where(tmp["opened"] > 0, tmp["clicked"] / tmp["opened"], np.nan)
        for _, row in tmp.drop(columns=["opened", "clicked"]).iterrows():
            rows.append({"feature": feature, **row.to_dict()})
    return pd.DataFrame(rows)


def make_recency_plot(recency_df: pd.DataFrame) -> Path:
    # recency 구간과 변수 표시 순서를 정의한다.
    bins = ["no prior", "0-1d", "2-3d", "4-7d", "7d+"]
    features = ["days_since_last_email", "days_since_last_open", "days_since_last_click"]

    # recency 구간별 Open Rate와 CTOR를 나란히 그린다.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for feature in features:
        part = recency_df[recency_df["feature"].eq(feature)].copy()
        part["bin"] = pd.Categorical(part["bin"], categories=bins, ordered=True)
        part = part.sort_values("bin")
        axes[0].plot(part["bin"].astype(str), part["open_rate"], marker="o", label=feature)
    axes[0].set_title("Open Rate by Recency", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Open Rate")
    axes[0].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for feature in features:
        part = recency_df[recency_df["feature"].eq(feature)].copy()
        part["bin"] = pd.Categorical(part["bin"], categories=bins, ordered=True)
        part = part.sort_values("bin")
        axes[1].plot(part["bin"].astype(str), part["ctor"], marker="o", label=feature)
    axes[1].set_title("CTOR by Recency", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("CTOR")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Recency Features and Response Rates", fontsize=16, fontweight="bold", y=1.03)
    return savefig("A_ml_eda_05_recency_rates.png")


def make_frequency_bins(df: pd.DataFrame) -> pd.DataFrame:
    # frequency 변수 목록을 정의한다.
    features = ["email_count_7d", "open_count_7d", "click_count_7d"]

    # 각 frequency 변수별 count 구간에 따른 반응률을 계산한다.
    rows = []
    for feature in features:
        part = df.copy()
        part["bin"] = pd.cut(
            part[feature],
            bins=[-0.001, 0, 1, 2, 3, 5, np.inf],
            labels=["0", "1", "2", "3", "4-5", "6+"],
            include_lowest=True,
        )
        tmp = (
            part.groupby("bin", observed=False)
            .agg(
                n=("target_opened", "size"),
                open_rate=("target_opened", "mean"),
                ctr=("target_clicked", "mean"),
                opened=("target_opened", "sum"),
                clicked=("target_clicked", "sum"),
            )
            .reset_index()
        )
        tmp["ctor"] = np.where(tmp["opened"] > 0, tmp["clicked"] / tmp["opened"], np.nan)
        for _, row in tmp.drop(columns=["opened", "clicked"]).iterrows():
            rows.append({"feature": feature, **row.to_dict()})
    return pd.DataFrame(rows)


def make_frequency_plot(freq_df: pd.DataFrame) -> Path:
    # frequency 구간과 변수 표시 순서를 정의한다.
    bins = ["0", "1", "2", "3", "4-5", "6+"]
    features = ["email_count_7d", "open_count_7d", "click_count_7d"]

    # frequency 구간별 Open Rate와 CTOR를 나란히 그린다.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for feature in features:
        part = freq_df[freq_df["feature"].eq(feature)].copy()
        part["bin"] = pd.Categorical(part["bin"].astype(str), categories=bins, ordered=True)
        part = part.sort_values("bin")
        axes[0].plot(part["bin"].astype(str), part["open_rate"], marker="o", label=feature)
    axes[0].set_title("Open Rate by 7d Frequency", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Open Rate")
    axes[0].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for feature in features:
        part = freq_df[freq_df["feature"].eq(feature)].copy()
        part["bin"] = pd.Categorical(part["bin"].astype(str), categories=bins, ordered=True)
        part = part.sort_values("bin")
        axes[1].plot(part["bin"].astype(str), part["ctor"], marker="o", label=feature)
    axes[1].set_title("CTOR by 7d Frequency", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("CTOR")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Frequency Features and Response Rates", fontsize=16, fontweight="bold", y=1.03)
    return savefig("A_ml_eda_06_frequency_rates.png")


def make_prior_summary(df: pd.DataFrame) -> pd.DataFrame:
    # prior flag별 반응률을 하나의 표로 합친다.
    rows = []
    for feature in ["has_prior_email", "has_prior_open", "has_prior_click"]:
        tmp = response_summary(df, feature)
        tmp.insert(0, "feature", feature)
        rows.append(tmp.rename(columns={feature: "value"}))
    return pd.concat(rows, ignore_index=True)


def make_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼별 결측 개수와 결측률을 계산한다.
    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing["missing_rate"] = missing["missing_count"] / len(df)
    return missing


def make_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    # recency/frequency 수치형 변수의 기초통계량을 계산한다.
    numeric_cols = [
        "days_since_last_email",
        "days_since_last_open",
        "days_since_last_click",
        "email_count_7d",
        "open_count_7d",
        "click_count_7d",
    ]
    stats = (
        df[numeric_cols]
        .describe(percentiles=[0.5, 0.9, 0.99])
        .T.reset_index()
        .rename(columns={"index": "variable"})
    )

    # 0보다 큰 값의 비율을 nonzero_rate로 계산한다.
    nonzero = (
        (df[numeric_cols].fillna(0) > 0)
        .mean()
        .rename("nonzero_rate")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    return stats.merge(nonzero, on="variable", how="left")


def make_overview(df: pd.DataFrame) -> pd.DataFrame:
    # 최종 데이터의 기간, 행/열 수, 주요 반응률을 한 행으로 정리한다.
    return pd.DataFrame(
        [
            {
                "rows": len(df),
                "columns": len(df.columns),
                "start_at": df["sent_at"].min(),
                "end_at": df["sent_at"].max(),
                "open_rate": df["target_opened"].mean(),
                "ctr": df["target_clicked"].mean(),
                "ctor": df.loc[df["target_opened"].eq(1), "target_clicked"].mean(),
            }
        ]
    )


def main() -> None:
    # 출력 폴더를 생성한다.
    ensure_output_dir()

    # 최종 전처리 데이터셋을 로드한다.
    df = pd.read_parquet(DATA_PATH)

    # 전체 데이터 개요와 결측/기초통계량을 계산한다.
    overview_df = make_overview(df)
    missing_df = make_missing_summary(df)
    numeric_df = make_numeric_summary(df)

    # 범주형 변수별 반응률 표를 계산한다.
    topic_df = response_summary(df, "topic").sort_values("ctor", ascending=False)
    provider_df = response_summary(df, "provider_group").sort_values("ctor", ascending=False)
    hour_df = response_summary(df, "send_hour").sort_values("send_hour")
    dow_df = response_summary(df, "send_dow").sort_values("send_dow")

    # 과거 이력 변수의 구간별 반응률 표를 계산한다.
    recency_df = make_recency_bins(df)
    freq_df = make_frequency_bins(df)
    prior_df = make_prior_summary(df)

    # 계산된 표를 CSV 파일로 저장한다.
    table_paths = [
        save_table(overview_df, "eda_01_overview.csv"),
        save_table(missing_df, "eda_02_missing.csv"),
        save_table(numeric_df, "eda_03_numeric_summary.csv"),
        save_table(topic_df, "eda_04_topic_rates.csv"),
        save_table(provider_df, "eda_05_provider_rates.csv"),
        save_table(hour_df, "eda_06_send_hour_rates.csv"),
        save_table(dow_df, "eda_07_send_dow_rates.csv"),
        save_table(recency_df, "eda_08_recency_rates.csv"),
        save_table(freq_df, "eda_09_frequency_rates.csv"),
        save_table(prior_df, "eda_10_prior_rates.csv"),
    ]

    # EDA 시각화 PNG 파일을 저장한다.
    figure_paths = [
        make_funnel_plot(df),
        make_topic_plot(topic_df),
        make_provider_plot(provider_df),
        make_send_time_plot(hour_df, dow_df),
        make_recency_plot(recency_df),
        make_frequency_plot(freq_df),
    ]

    # 콘솔에서 주요 결과와 저장된 파일 경로를 확인한다.
    print("\n[Overview]")
    print(overview_df.to_string(index=False))
    print("\n[Saved tables]")
    for path in table_paths:
        print(path.resolve())
    print("\n[Saved figures]")
    for path in figure_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
