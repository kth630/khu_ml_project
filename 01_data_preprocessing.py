"""Trigger Email 모델링용 최종 분석 데이터를 생성하는 전처리 스크립트.

실행 결과
---------
- 원본 메시지 로그에서 email 채널 + trigger 캠페인만 추출한다.
- 고객별 과거 발송/open/click 이력 기반 recency/frequency feature를 만든다.
- 최종 모델링 데이터는 outputs/processed/A_ml_dataset.pkl에 저장한다.

주의
----
- messages-demo.csv가 매우 크기 때문에 chunk 단위로 읽는다.
- pyarrow가 없는 환경에서도 실행되도록 parquet 대신 pickle로 저장한다.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# 모든 입력/출력 경로는 이 스크립트가 있는 email_ml 폴더를 기준으로 잡는다.
ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
PROCESSED_DIR = OUTPUT_ROOT / "processed"

INPUT_MESSAGES = ROOT / "messages-demo.csv"
INPUT_CAMPAIGNS = ROOT / "campaigns.csv"
INPUT_CLIENTS = ROOT / "client_first_purchase_date.csv"
INPUT_HOLIDAYS = ROOT / "holidays.csv"

OUTPUT_DATASET = PROCESSED_DIR / "A_ml_dataset.pkl"
OUTPUT_SAMPLE = PROCESSED_DIR / "A_ml_dataset_sample.csv"
OUTPUT_SUMMARY_CSV = PROCESSED_DIR / "preprocessing_summary.csv"
OUTPUT_SUMMARY_JSON = PROCESSED_DIR / "preprocessing_summary.json"

CHUNKSIZE = 500_000
PROVIDER_MIN_COUNT = 10_000

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


def ensure_output_dir() -> None:
    """전처리 산출물을 저장할 폴더를 만든다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def check_inputs() -> None:
    """전처리에 반드시 필요한 원본 CSV 파일이 있는지 확인한다."""
    required = [INPUT_MESSAGES, INPUT_CAMPAIGNS]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(str(path) for path in missing))


def as_bool(series: pd.Series) -> pd.Series:
    """t/f, true/false, 1/0 형태가 섞인 값을 boolean으로 변환한다."""
    return series.astype("string").str.lower().isin(["true", "t", "1", "yes", "y"])


def load_trigger_email_messages() -> tuple[pd.DataFrame, dict[str, int]]:
    """email 채널이면서 trigger 캠페인에 속한 메시지만 읽어온다."""
    campaigns = pd.read_csv(INPUT_CAMPAIGNS, usecols=["id", "campaign_type", "topic"])
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
        "message_id": "string",
        "client_id": "string",
        "channel": "string",
        "email_provider": "string",
        "is_opened": "string",
        "is_clicked": "string",
        "is_hard_bounced": "string",
        "is_soft_bounced": "string",
    }

    stats = {
        "message_rows_read": 0,
        "email_rows": 0,
        "trigger_email_rows": 0,
        "trigger_campaigns": int(len(trigger_campaigns)),
    }
    parts: list[pd.DataFrame] = []

    # 대용량 messages-demo.csv를 한 번에 메모리에 올리지 않기 위해 chunk 단위로 처리한다.
    for chunk_no, chunk in enumerate(
        pd.read_csv(INPUT_MESSAGES, usecols=usecols, dtype=dtype, chunksize=CHUNKSIZE),
        start=1,
    ):
        stats["message_rows_read"] += int(len(chunk))
        chunk = chunk[chunk["channel"].eq("email")]
        stats["email_rows"] += int(len(chunk))
        chunk = chunk.merge(trigger_campaigns, on="campaign_id", how="inner")
        stats["trigger_email_rows"] += int(len(chunk))
        if len(chunk):
            parts.append(chunk)
        print(
            f"   chunk {chunk_no}: read={stats['message_rows_read']:,}, "
            f"email={stats['email_rows']:,}, trigger_email={stats['trigger_email_rows']:,}"
        )

    if not parts:
        raise ValueError("No email trigger messages were found after filtering.")

    df = pd.concat(parts, ignore_index=True)
    df["sent_at"] = pd.to_datetime(df["sent_at"], errors="coerce")
    df = df[df["sent_at"].notna()].copy()

    for col in ["is_opened", "is_clicked", "is_hard_bounced", "is_soft_bounced"]:
        df[col] = as_bool(df[col]).astype(np.int8)

    df["client_id"] = df["client_id"].astype(str)
    df["topic"] = df["topic"].fillna("unknown").astype(str)
    df["email_provider"] = df["email_provider"].fillna("unknown").astype(str)
    return df, stats


def build_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """고객별 과거 발송/open/click 이력을 이용해 history feature를 생성한다."""
    df = df.sort_values(["client_id", "sent_at", "campaign_id", "message_id"]).reset_index(drop=True)
    n = len(df)

    sent_ns = df["sent_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
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

    # 각 고객의 마지막 행동 시각과 최근 7일 이벤트 큐를 갱신하면서 feature를 만든다.
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

        if i and i % 500_000 == 0:
            print(f"   history rows processed: {i:,}/{n:,}")

    features = pd.DataFrame(feats)
    return pd.concat([df.reset_index(drop=True), features], axis=1)


def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """warm-up 기간 제거, 이상 행 제거, provider grouping 후 최종 컬럼을 선택한다."""
    cutoff = df["sent_at"].min() + pd.Timedelta(days=7)
    out = df[df["sent_at"].ge(cutoff)].copy()
    out = out[~out["topic"].eq("price drop")].copy()

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
    return final.sort_values("sent_at").reset_index(drop=True)


def write_summary(final: pd.DataFrame, stats: dict[str, int]) -> None:
    """전처리 과정에서 확인한 핵심 수치를 CSV와 JSON으로 저장한다."""
    opened = final[final["target_opened"].eq(1)]
    summary = {
        **stats,
        "final_rows": int(len(final)),
        "final_columns": int(len(final.columns)),
        "period_start": str(final["sent_at"].min()),
        "period_end": str(final["sent_at"].max()),
        "open_rate": float(final["target_opened"].mean()),
        "ctr": float(final["target_clicked"].mean()),
        "ctor": float(opened["target_clicked"].mean()) if len(opened) else None,
        "input_messages": str(INPUT_MESSAGES),
        "input_campaigns": str(INPUT_CAMPAIGNS),
        "optional_client_first_purchase_date_exists": INPUT_CLIENTS.exists(),
        "optional_holidays_exists": INPUT_HOLIDAYS.exists(),
        "output_dataset": str(OUTPUT_DATASET),
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """전처리 전체 pipeline을 순서대로 실행한다."""
    ensure_output_dir()
    check_inputs()

    print("1/4 Load email trigger messages")
    raw, stats = load_trigger_email_messages()
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

    print("4/4 Save processed artifacts")
    final.to_pickle(OUTPUT_DATASET)
    final.head(10_000).to_csv(OUTPUT_SAMPLE, index=False, encoding="utf-8-sig")
    write_summary(final, stats)
    print(f"Saved dataset: {OUTPUT_DATASET.resolve()}")
    print(f"Saved summary: {OUTPUT_SUMMARY_CSV.resolve()}")


if __name__ == "__main__":
    main()
