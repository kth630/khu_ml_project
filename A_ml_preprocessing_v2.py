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


PROJECT_ROOT = resolve_project_root(["messages-demo.csv", "campaigns.csv"])
INPUT_MESSAGES = PROJECT_ROOT / "messages-demo.csv"
INPUT_CAMPAIGNS = PROJECT_ROOT / "campaigns.csv"
OUTPUT_DATASET = PROJECT_ROOT / "A_ml_dataset.parquet"


def as_bool(series: pd.Series) -> pd.Series:
    # 문자열/숫자/boolean이 섞여 있을 수 있는 컬럼을 True/False 값으로 변환한다.
    return series.astype(str).str.lower().isin(["true", "t", "1"])


def load_trigger_email_messages() -> pd.DataFrame:
    # 캠페인 데이터에서 trigger 캠페인의 id와 topic만 가져온다.
    campaigns = pd.read_csv(
        INPUT_CAMPAIGNS,
        usecols=["id", "campaign_type", "topic"],
    )
    trigger_campaigns = (
        campaigns[campaigns["campaign_type"].eq("trigger")][["id", "topic"]]
        .rename(columns={"id": "campaign_id"})
        .copy()
    )

    # 메시지 데이터에서 최종 분석에 필요한 컬럼만 읽는다.
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

    # boolean 반응 컬럼은 원본에서 True/False와 문자열이 섞일 수 있어 문자열로 읽는다.
    dtype = {
        "is_opened": "string",
        "is_clicked": "string",
        "is_hard_bounced": "string",
        "is_soft_bounced": "string",
    }

    # messages-demo.csv가 크기 때문에 chunk 단위로 읽어서 email + trigger만 남긴다.
    parts = []
    for chunk in pd.read_csv(INPUT_MESSAGES, usecols=usecols, dtype=dtype, chunksize=CHUNKSIZE):
        # email 채널 메시지만 남긴다.
        chunk = chunk[chunk["channel"].eq("email")]

        # trigger 캠페인 목록과 inner join하여 email-trigger 메시지만 남긴다.
        chunk = chunk.merge(trigger_campaigns, on="campaign_id", how="inner")
        if len(chunk):
            parts.append(chunk)

    # chunk별로 필터링된 데이터를 하나의 DataFrame으로 합친다.
    df = pd.concat(parts, ignore_index=True)

    # 발송 시각을 datetime 타입으로 변환하고 변환 실패 행은 제거한다.
    df["sent_at"] = pd.to_datetime(df["sent_at"], errors="coerce")
    df = df[df["sent_at"].notna()].copy()

    # 반응/바운스 컬럼을 0/1 정수형으로 변환한다.
    for col in ["is_opened", "is_clicked", "is_hard_bounced", "is_soft_bounced"]:
        df[col] = as_bool(df[col]).astype(np.int8)

    # 조인 및 그룹화에 사용할 문자열 컬럼들의 타입과 결측을 정리한다.
    df["client_id"] = df["client_id"].astype(str)
    df["topic"] = df["topic"].astype(str)
    df["email_provider"] = df["email_provider"].fillna("unknown").astype(str)
    return df


def build_history_features(df: pd.DataFrame) -> pd.DataFrame:
    # 고객별 과거 이력을 계산하기 위해 고객 id와 발송 시각 순서로 정렬한다.
    df = df.sort_values(["client_id", "sent_at", "campaign_id", "message_id"]).reset_index(drop=True)
    n = len(df)

    # 반복문에서 빠르게 접근할 수 있도록 필요한 컬럼을 numpy 배열로 변환한다.
    sent_ns = df["sent_at"].astype("int64").to_numpy()
    client = df["client_id"].to_numpy()
    opened = df["is_opened"].to_numpy()
    clicked = df["is_clicked"].to_numpy()

    # 시간 차이를 일 단위로 계산하기 위한 nanosecond 상수를 정의한다.
    one_day_ns = 24 * 60 * 60 * 1_000_000_000
    window_7d = 7 * one_day_ns
    events = ["email", "open", "click"]

    # 마지막 수신/오픈/클릭 이후 경과일 컬럼명을 정의한다.
    recency_cols = [
        "days_since_last_email",
        "days_since_last_open",
        "days_since_last_click",
    ]

    # 과거 수신/오픈/클릭 경험 여부 flag 컬럼명을 정의한다.
    flag_cols = [
        "has_prior_email",
        "has_prior_open",
        "has_prior_click",
    ]

    # 최근 7일 수신/오픈/클릭 횟수 컬럼명을 정의한다.
    count_cols = [f"{event}_count_7d" for event in events]

    # recency는 과거 행동이 없으면 NaN으로 둔다.
    feats = {col: np.full(n, np.nan, dtype=np.float32) for col in recency_cols}

    # prior flag는 과거 행동이 없으면 0, 있으면 1로 채운다.
    feats.update({col: np.zeros(n, dtype=np.int8) for col in flag_cols})

    # 최근 7일 count 변수는 기본값 0으로 시작한다.
    feats.update({col: np.zeros(n, dtype=np.int16) for col in count_cols})

    # 고객별 마지막 email/open/click 발생 시각을 저장한다.
    last = defaultdict(dict)

    # 고객별 최근 7일 window 안의 email/open/click 발생 시각을 deque로 저장한다.
    overall = defaultdict(lambda: {event: deque() for event in events})

    for i in range(n):
        c = client[i]
        ts = int(sent_ns[i])

        # 현재 발송 이전 마지막 이벤트 시각을 이용해 recency와 prior flag를 채운다.
        for event, rec_col, flag_col in [
            ("email", "days_since_last_email", "has_prior_email"),
            ("open", "days_since_last_open", "has_prior_open"),
            ("click", "days_since_last_click", "has_prior_click"),
        ]:
            if event in last[c]:
                feats[rec_col][i] = (ts - last[c][event]) / one_day_ns
                feats[flag_col][i] = 1

        # 현재 발송 시각 기준 7일 이전보다 오래된 이벤트를 window에서 제거한다.
        cutoff = ts - window_7d
        for event in events:
            q = overall[c][event]
            while q and q[0] < cutoff:
                q.popleft()
            feats[f"{event}_count_7d"][i] = len(q)

        # 현재 행의 결과는 현재 행 feature 계산 이후에 추가한다.
        event_flags = {
            "email": True,
            "open": opened[i] == 1,
            "click": clicked[i] == 1,
        }
        for event, should_add in event_flags.items():
            if not should_add:
                continue
            overall[c][event].append(ts)
            last[c][event] = ts

    # 계산한 과거 이력 feature를 원본 행과 옆으로 합친다.
    features = pd.DataFrame(feats)
    return pd.concat([df.reset_index(drop=True), features], axis=1)


def prepare_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # 초기 7일은 과거 이력 window를 채우기 위한 구간으로 사용하고 최종 데이터에서 제외한다.
    cutoff = df["sent_at"].min() + pd.Timedelta(days=7)
    out = df[df["sent_at"].ge(cutoff)].copy()

    # 표본 수가 극히 작은 price drop topic을 제거한다.
    out = out[~out["topic"].eq("price drop")].copy()

    # hard/soft bounce가 발생했는데 opened로 기록된 행은 반응 해석에서 제외한다.
    bounce_open = (
        out["is_opened"].eq(1)
        & (out["is_hard_bounced"].eq(1) | out["is_soft_bounced"].eq(1))
    )
    out = out[~bounce_open].copy()

    # provider별 표본 수를 계산하고 기준 미만 provider는 other로 묶는다.
    provider_counts = out["email_provider"].value_counts()
    keep_providers = set(provider_counts[provider_counts >= PROVIDER_MIN_COUNT].index)
    out["provider_group"] = np.where(
        out["email_provider"].isin(keep_providers),
        out["email_provider"],
        "other",
    )

    # 발송 시각에서 시간과 요일을 추출하고 범주형 변수로 쓰기 위해 문자열로 저장한다.
    out["send_hour"] = out["sent_at"].dt.hour.astype(str)
    out["send_dow"] = out["sent_at"].dt.dayofweek.astype(str)

    # 최종 데이터에 남길 기본 식별/타깃/범주형 컬럼을 정의한다.
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

    # 최종 데이터에 남길 과거 이력 feature 컬럼을 정의한다.
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

    # 최종 컬럼만 선택한다.
    final = out[base_cols + numeric_feature_cols].copy()

    # 모델링용 타깃 컬럼명을 target_* 형태로 변경한다.
    final = final.rename(
        columns={
            "is_opened": "target_opened",
            "is_clicked": "target_clicked",
        }
    )

    # 최종 데이터는 전체 시간 순서대로 정렬한다.
    final = final.sort_values("sent_at").reset_index(drop=True)
    return final


def main() -> None:
    # 1단계: 원본 messages/campaigns에서 email-trigger 메시지를 로드한다.
    print("1/4 Load email trigger messages")
    raw = load_trigger_email_messages()
    print(f"   raw trigger email rows: {len(raw):,}")

    # 2단계: 현재 발송 이전 정보만 사용해 recency/frequency feature를 만든다.
    print("2/4 Build recency/frequency features")
    featured = build_history_features(raw)

    # 3단계: warm-up 제외, 논리 오류 제거, provider grouping, 최종 컬럼 선택을 수행한다.
    print("3/4 Apply final preprocessing")
    final = prepare_final_dataset(featured)
    print(f"   final rows: {len(final):,}")
    print(f"   final columns: {len(final.columns):,}")
    print(f"   open rate: {final['target_opened'].mean():.4f}")
    opened = final[final["target_opened"].eq(1)]
    print(f"   click-after-open rate: {opened['target_clicked'].mean():.4f}")

    # 4단계: 최종 모델링 데이터셋을 parquet 파일로 저장한다.
    print("4/4 Save A_ml_dataset.parquet")
    final.to_parquet(OUTPUT_DATASET, index=False)
    print(f"Saved: {OUTPUT_DATASET.resolve()}")


if __name__ == "__main__":
    main()
