"""전처리된 Trigger Email 데이터의 EDA 표와 그림을 생성하는 스크립트.

실행 결과
---------
- Funnel, topic, provider, send time, recency, frequency 요약표를 저장한다.
- 보고서에 사용할 PNG 그래프를 outputs/eda 폴더에 저장한다.

주의
----
- 입력 파일은 outputs/processed/A_ml_dataset.pkl이다.
- EDA 산출물은 보고서용 자료이므로 같은 이름으로 다시 저장하면 기존 파일을 덮어쓴다.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 모든 경로는 email_ml 폴더를 기준으로 고정한다.
ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "outputs" / "processed"
EDA_DIR = ROOT / "outputs" / "eda"

DATA_PATH = PROCESSED_DIR / "A_ml_dataset.pkl"
PROFILE_PATH = EDA_DIR / "eda_profile.json"

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir() -> None:
    """EDA 산출물 폴더를 만들고 기존 EDA 파일을 정리한다."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    for path in EDA_DIR.iterdir():
        if path.is_file():
            path.unlink()


def savefig(name: str) -> Path:
    """현재 matplotlib figure를 PNG 파일로 저장한다."""
    path = EDA_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    """EDA 요약 DataFrame을 CSV 파일로 저장한다."""
    path = EDA_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def response_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """특정 범주별 발송 수, open rate, CTR, CTOR를 계산한다."""
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
    grouped["ctor"] = np.where(grouped["opened"] > 0, grouped["clicked"] / grouped["opened"], np.nan)
    return grouped.drop(columns=["opened", "clicked"])


def make_funnel_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sent -> Open -> Click funnel 요약표를 만든다."""
    sent = len(df)
    opened = int(df["target_opened"].sum())
    clicked = int(df["target_clicked"].sum())
    return pd.DataFrame(
        [
            {"stage": "Sent", "count": sent, "rate_vs_sent": 1.0, "rate_vs_previous": 1.0},
            {"stage": "Open", "count": opened, "rate_vs_sent": opened / sent, "rate_vs_previous": opened / sent},
            {
                "stage": "Click",
                "count": clicked,
                "rate_vs_sent": clicked / sent,
                "rate_vs_previous": clicked / opened if opened else np.nan,
            },
        ]
    )


def make_funnel_plot(funnel: pd.DataFrame) -> Path:
    """funnel 단계별 count와 sent 대비 비율을 시각화한다."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot_df = funnel.iloc[::-1]
    bars = ax.barh(plot_df["stage"], plot_df["count"])
    ax.set_title("Trigger Email Funnel", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of messages")
    ax.grid(axis="x", alpha=0.25)
    max_count = funnel["count"].max()
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + max_count * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{row['rate_vs_sent']:.2%}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlim(0, max_count * 1.28)
    return savefig("A_ml_eda_01_funnel.png")


def make_rate_bar_plot(summary: pd.DataFrame, group_col: str, filename: str, title: str) -> Path:
    """범주별 Open Rate와 CTOR를 가로 막대그래프로 저장한다."""
    plot_df = summary.sort_values("ctor", ascending=True)
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(10, max(4.8, len(plot_df) * 0.45)))
    ax.barh(y - 0.18, plot_df["open_rate"], height=0.36, label="Open Rate")
    ax.barh(y + 0.18, plot_df["ctor"], height=0.36, label="CTOR")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[group_col].astype(str))
    ax.set_xlabel("Response rate")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    return savefig(filename)


def make_send_time_plot(hour_df: pd.DataFrame, dow_df: pd.DataFrame) -> Path:
    """발송 시간대와 요일별 반응률 변화를 함께 시각화한다."""
    hour_df = hour_df.copy()
    hour_df["send_hour_num"] = hour_df["send_hour"].astype(int)
    hour_df = hour_df.sort_values("send_hour_num")

    dow_df = dow_df.copy()
    dow_df["send_dow_num"] = dow_df["send_dow"].astype(int)
    dow_df = dow_df.sort_values("send_dow_num")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(hour_df["send_hour_num"], hour_df["open_rate"], marker="o", label="Open Rate")
    axes[0].plot(hour_df["send_hour_num"], hour_df["ctor"], marker="o", label="CTOR")
    axes[0].set_title("Response Rate by Send Hour", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("send_hour")
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
    """최근 발송/open/click 이후 경과일을 구간화해 반응률을 요약한다."""
    features = ["days_since_last_email", "days_since_last_open", "days_since_last_click"]
    flags = {
        "days_since_last_email": "has_prior_email",
        "days_since_last_open": "has_prior_open",
        "days_since_last_click": "has_prior_click",
    }
    rows = []
    for feature in features:
        flag = flags[feature]
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
        part = df[df[flag].eq(1)].copy()
        part["bin"] = pd.cut(
            part[feature],
            bins=[-0.001, 1, 3, 7, np.inf],
            labels=["0-1d", "2-3d", "4-7d", "7d+"],
            include_lowest=True,
        )
        tmp = response_summary(part, "bin")
        for _, row in tmp.iterrows():
            rows.append({"feature": feature, **row.to_dict()})
    return pd.DataFrame(rows)


def make_frequency_bins(df: pd.DataFrame) -> pd.DataFrame:
    """최근 7일 발송/open/click 횟수를 구간화해 반응률을 요약한다."""
    features = ["email_count_7d", "open_count_7d", "click_count_7d"]
    rows = []
    for feature in features:
        part = df.copy()
        part["bin"] = pd.cut(
            part[feature],
            bins=[-0.001, 0, 1, 2, 3, 5, np.inf],
            labels=["0", "1", "2", "3", "4-5", "6+"],
            include_lowest=True,
        )
        tmp = response_summary(part, "bin")
        for _, row in tmp.iterrows():
            rows.append({"feature": feature, **row.to_dict()})
    return pd.DataFrame(rows)


def make_feature_bin_plot(summary: pd.DataFrame, filename: str, title_prefix: str) -> Path:
    """recency/frequency 구간별 Open Rate와 CTOR를 선 그래프로 저장한다."""
    features = summary["feature"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for feature in features:
        part = summary[summary["feature"].eq(feature)].copy()
        axes[0].plot(part["bin"].astype(str), part["open_rate"], marker="o", label=feature)
        axes[1].plot(part["bin"].astype(str), part["ctor"], marker="o", label=feature)
    axes[0].set_title(f"Open Rate by {title_prefix}", fontsize=13, fontweight="bold")
    axes[1].set_title(f"CTOR by {title_prefix}", fontsize=13, fontweight="bold")
    for ax in axes:
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    return savefig(filename)


def main() -> None:
    """EDA 표와 그래프를 순서대로 생성한다."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:.6f}".format)

    ensure_output_dir()

    print("1/8 Load final dataset")
    df = pd.read_pickle(DATA_PATH)
    print(f"   rows: {len(df):,}")
    print(f"   period: {df['sent_at'].min()} ~ {df['sent_at'].max()}")

    print("2/8 Funnel summary")
    funnel = make_funnel_table(df)
    save_table(funnel, "A_ml_eda_01_funnel_summary.csv")
    make_funnel_plot(funnel)

    print("3/8 Topic summary")
    topic_summary = response_summary(df, "topic")
    save_table(topic_summary, "A_ml_eda_02_topic_summary.csv")
    make_rate_bar_plot(topic_summary, "topic", "A_ml_eda_02_topic_rates.png", "Topic Open Rate / CTOR")

    print("4/8 Provider summary")
    provider_summary = response_summary(df, "provider_group")
    save_table(provider_summary, "A_ml_eda_03_provider_summary.csv")
    make_rate_bar_plot(provider_summary, "provider_group", "A_ml_eda_03_provider_rates.png", "Provider Open Rate / CTOR")

    print("5/8 Send time summary")
    hour_summary = response_summary(df, "send_hour")
    dow_summary = response_summary(df, "send_dow")
    save_table(hour_summary, "A_ml_eda_04_send_hour_summary.csv")
    save_table(dow_summary, "A_ml_eda_05_send_dow_summary.csv")
    make_send_time_plot(hour_summary, dow_summary)

    print("6/8 Recency summary")
    recency_summary = make_recency_bins(df)
    save_table(recency_summary, "A_ml_eda_06_recency_summary.csv")
    make_feature_bin_plot(recency_summary, "A_ml_eda_05_recency_rates.png", "Recency")

    print("7/8 Frequency summary")
    frequency_summary = make_frequency_bins(df)
    save_table(frequency_summary, "A_ml_eda_07_frequency_summary.csv")
    make_feature_bin_plot(frequency_summary, "A_ml_eda_06_frequency_rates.png", "Frequency")

    print("8/8 Save basic dataset profile")
    profile = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "start_at": str(df["sent_at"].min()),
        "end_at": str(df["sent_at"].max()),
        "open_rate": float(df["target_opened"].mean()),
        "ctr": float(df["target_clicked"].mean()),
        "ctor": float(df.loc[df["target_opened"].eq(1), "target_clicked"].mean()),
        "output_dir": str(EDA_DIR),
    }
    save_table(pd.DataFrame([profile]), "A_ml_eda_00_dataset_profile.csv")
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved outputs: {EDA_DIR.resolve()}")


if __name__ == "__main__":
    main()
