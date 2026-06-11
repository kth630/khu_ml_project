from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_project_root(required_files: list[str]) -> Path:
    """Find the project root that contains all required files."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    known_project_dir = (
        Path.home()
        / "Desktop"
        / "학교"
        / "26-1"
        / "머신러닝기초및응용"
        / "머신러닝_프젝"
        / "ecommerce"
    )

    candidates = [script_dir, cwd, known_project_dir]
    candidates.extend(script_dir.parents)
    candidates.extend(cwd.parents)

    for candidate in candidates:
        if all((candidate / name).exists() for name in required_files):
            return candidate

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Required project files were not found.\n"
        f"Required files: {required_files}\n"
        f"Checked paths:\n{checked}"
    )


ROOT = resolve_project_root(["A_ml_dataset.parquet"])
DATA_PATH = ROOT / "A_ml_dataset.parquet"
OUTPUT_DIR = ROOT / "A_ml_eda_outputs"

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def savefig(name: str) -> Path:
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def response_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
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


def make_frequency_bins(df: pd.DataFrame) -> pd.DataFrame:
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


def make_feature_bin_plot(summary: pd.DataFrame, filename: str, title_prefix: str) -> Path:
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
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:.6f}".format)

    ensure_output_dir()

    print("1/8 Load final dataset")
    df = pd.read_parquet(DATA_PATH)
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
    profile = pd.DataFrame(
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
    save_table(profile, "A_ml_eda_00_dataset_profile.csv")

    print(f"Saved outputs: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
