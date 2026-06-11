from collections import defaultdict, deque
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


CHUNKSIZE = 500_000
PROVIDER_MIN_COUNT = 10_000

# 원본 CSV 일부 컬럼의 mixed type 경고를 숨긴다.
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


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


PROJECT_ROOT = resolve_project_root(["messages-demo.csv", "campaigns.csv"])
INPUT_MESSAGES = PROJECT_ROOT / "messages-demo.csv"
INPUT_CAMPAIGNS = PROJECT_ROOT / "campaigns.csv"
OUTPUT_DATASET = PROJECT_ROOT / "A_ml_dataset.parquet"


def as_bool(series: pd.Series) -> pd.Series:
    """Convert mixed boolean-like values into True/False."""
    return series.astype(str).str.lower().isin(["true", "t", "1"])


def load_trigger_email_messages() -> pd.DataFrame:
    """Load email-channel messages that belong to trigger campaigns."""
    campaigns = pd.read_csv(
        INPUT_CAMPAIGNS,
        usecols=["id", "campaign_type", "topic"],
    )
    trigger_campaigns = (
        campaigns[campaigns["campaign_type"].eq("trigger")][["id", "topic"]]
        .rename(columns={"id": "campaign_id"})
        .copy()
    )

    usecols = [
        "campaign_id",
        "message_id",
        "client_id",
        "channel",
        "email_provider",
        "sent_at",
        "is_opened",
        "is_clicked",
        "is_hard_bounced",
        "is_soft_bounced",
    ]
    dtype = {
        "is_opened": "string",
        "is_clicked": "string",
        "is_hard_bounced": "string",
        "is_soft_bounced": "string",
    }

    parts = []
    for chunk in pd.read_csv(INPUT_MESSAGES, usecols=usecols, dtype=dtype, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["channel"].eq("email")]
        chunk = chunk.merge(trigger_campaigns, on="campaign_id", how="inner")
        if len(chunk):
            parts.append(chunk)

    if not parts:
        raise ValueError("No email trigger messages were found after filtering.")

    df = pd.concat(parts, ignore_index=True)
    df["sent_at"] = pd.to_datetime(df["sent_at"], errors="coerce")
    df = df[df["sent_at"].notna()].copy()

    for col in ["is_opened", "is_clicked", "is_hard_bounced", "is_soft_bounced"]:
        df[col] = as_bool(df[col]).astype(np.int8)

    df["client_id"] = df["client_id"].astype(str)
    df["topic"] = df["topic"].astype(str)
    df["email_provider"] = df["email_provider"].fillna("unknown").astype(str)
    return df


def build_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create prior-history, recency, and 7-day frequency features per client."""
    df = df.sort_values(["client_id", "sent_at", "campaign_id", "message_id"]).reset_index(drop=True)
    n = len(df)

    sent_ns = df["sent_at"].astype("int64").to_numpy()
    client = df["client_id"].to_numpy()
    opened = df["is_opened"].to_numpy()
    clicked = df["is_clicked"].to_numpy()

    one_day_ns = 24 * 60 * 60 * 1_000_000_000
    window_7d = 7 * one_day_ns
    events = ["email", "open", "click"]

    recency_cols = [
        "days_since_last_email",
        "days_since_last_open",
        "days_since_last_click",
    ]
    flag_cols = [
        "has_prior_email",
        "has_prior_open",
        "has_prior_click",
    ]
    count_cols = [f"{event}_count_7d" for event in events]

    feats = {col: np.full(n, np.nan, dtype=np.float32) for col in recency_cols}
    feats.update({col: np.zeros(n, dtype=np.int8) for col in flag_cols})
    feats.update({col: np.zeros(n, dtype=np.int16) for col in count_cols})

    last = defaultdict(dict)
    recent_events = defaultdict(lambda: {event: deque() for event in events})

    for i in range(n):
        c = client[i]
        ts = int(sent_ns[i])

        for event, rec_col, flag_col in [
            ("email", "days_since_last_email", "has_prior_email"),
            ("open", "days_since_last_open", "has_prior_open"),
            ("click", "days_since_last_click", "has_prior_click"),
        ]:
            if event in last[c]:
                feats[rec_col][i] = (ts - last[c][event]) / one_day_ns
                feats[flag_col][i] = 1

        cutoff = ts - window_7d
        for event in events:
            q = recent_events[c][event]
            while q and q[0] < cutoff:
                q.popleft()
            feats[f"{event}_count_7d"][i] = len(q)

        event_flags = {
            "email": True,
            "open": opened[i] == 1,
            "click": clicked[i] == 1,
        }
        for event, should_add in event_flags.items():
            if should_add:
                recent_events[c][event].append(ts)
                last[c][event] = ts

    features = pd.DataFrame(feats)
    return pd.concat([df.reset_index(drop=True), features], axis=1)


def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply warm-up exclusion, error filtering, provider grouping, and final column selection."""
    cutoff = df["sent_at"].min() + pd.Timedelta(days=7)
    out = df[df["sent_at"].ge(cutoff)].copy()

    # 표본 수가 극히 작은 topic은 안정적인 모델링과 해석을 위해 제외한다.
    out = out[~out["topic"].eq("price drop")].copy()

    # 바운스 처리된 메시지가 opened로 기록된 논리 오류 행은 제외한다.
    bounce_open = (
        out["is_opened"].eq(1)
        & (out["is_hard_bounced"].eq(1) | out["is_soft_bounced"].eq(1))
    )
    out = out[~bounce_open].copy()

    provider_counts = out["email_provider"].value_counts()
    keep_providers = set(provider_counts[provider_counts >= PROVIDER_MIN_COUNT].index)
    out["provider_group"] = np.where(
        out["email_provider"].isin(keep_providers),
        out["email_provider"],
        "other",
    )

    out["send_hour"] = out["sent_at"].dt.hour.astype(str)
    out["send_dow"] = out["sent_at"].dt.dayofweek.astype(str)

    base_cols = [
        "message_id",
        "campaign_id",
        "client_id",
        "sent_at",
        "is_opened",
        "is_clicked",
        "topic",
        "provider_group",
        "send_hour",
        "send_dow",
    ]
    numeric_feature_cols = [
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

    final = out[base_cols + numeric_feature_cols].copy()
    final = final.rename(
        columns={
            "is_opened": "target_opened",
            "is_clicked": "target_clicked",
        }
    )
    final = final.sort_values("sent_at").reset_index(drop=True)
    return final


def main() -> None:
    print("1/4 Load email trigger messages")
    raw = load_trigger_email_messages()
    print(f"   raw trigger email rows: {len(raw):,}")

    print("2/4 Build recency/frequency features")
    featured = build_history_features(raw)

    print("3/4 Apply final preprocessing")
    final = prepare_final_dataset(featured)
    print(f"   final rows: {len(final):,}")
    print(f"   final columns: {len(final.columns):,}")
    print(f"   open rate: {final['target_opened'].mean():.4f}")
    opened = final[final["target_opened"].eq(1)]
    print(f"   click-after-open rate: {opened['target_clicked'].mean():.4f}")

    print("4/4 Save A_ml_dataset.parquet")
    final.to_parquet(OUTPUT_DATASET, index=False)
    print(f"Saved: {OUTPUT_DATASET.resolve()}")


if __name__ == "__main__":
    main()
