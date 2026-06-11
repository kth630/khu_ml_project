# Trigger Email Funnel Analysis

이 저장소는 이메일 마케팅 데이터를 이용해 Trigger Email 반응을 단계별로 예측하고 해석하는 머신러닝 프로젝트 코드입니다.

핵심 목표는 CTR을 하나의 결과로만 보지 않고, 다음 두 단계로 분리해 분석하는 것입니다.

```text
Sent -> Open -> Click
```

- Open Model: 전체 발송 건 중 이메일 오픈 여부 예측
- Click-after-Open Model: 이메일을 오픈한 고객 중 클릭 여부 예측, 즉 CTOR 관점의 후속 모델

## 파일 구성

```text
01_data_preprocessing.py       # 보고서용 전처리 실행 파일
02_exploratory_analysis.py     # 보고서용 EDA 실행 파일
03_modeling_and_evaluation.py  # 모델 비교, threshold tuning, 최종 LightGBM XAI 실행 파일

A_ml_preprocessing.py          # 기존 전처리 원본 코드
A_ml_eda.py                    # 기존 EDA 원본 코드
A_ml_modeling_v2.py            # 기존 최신 LightGBM + threshold tuning + XAI 코드

requirements.txt               # 실행에 필요한 패키지 목록
```

`01_data_preprocessing.py`와 `02_exploratory_analysis.py`는 기존 원본 코드를 삭제하지 않고, 보고서 제출용 실행 순서를 명확히 하기 위해 만든 entry point입니다.

`03_modeling_and_evaluation.py`는 보고서 피드백을 반영해 모델 성능 비교를 추가한 파일입니다. 모델 비교에서는 Logistic Regression을 제외하고, Random Forest, XGBoost, LightGBM을 비교합니다. 이후 최종 해석은 기존 LightGBM 기반 XAI 파이프라인을 사용합니다.

## 입력 파일

프로젝트 루트 폴더에 아래 파일이 필요합니다.

```text
messages-demo.csv
campaigns.csv
```

전처리 후 다음 파일이 생성됩니다.

```text
A_ml_dataset.parquet
```

## 실행 방법

```bash
pip install -r requirements.txt
python 01_data_preprocessing.py
python 02_exploratory_analysis.py
python 03_modeling_and_evaluation.py
```

기존 파일을 직접 실행할 수도 있지만, 보고서 기준 실행 순서는 위의 `01 -> 02 -> 03` 구조를 권장합니다.

## 분석 흐름

1. trigger 캠페인과 email 채널 메시지를 필터링합니다.
2. 고객별 과거 이메일 수신, 오픈, 클릭 이력을 생성합니다.
3. 발송 시각, 주제, 이메일 제공자, 과거 반응 이력 변수를 사용해 모델 입력 변수를 구성합니다.
4. 시간 순서를 기준으로 train, validation, test set을 분리합니다.
5. Open Model과 Click-after-Open Model을 각각 학습합니다.
6. Random Forest, XGBoost, LightGBM을 동일한 split과 동일한 입력 변수 기준으로 비교합니다.
7. validation set에서 threshold별 Precision, Recall, F1-score를 비교하고 최종 threshold를 선택합니다.
8. 선택한 threshold를 기준으로 validation/test 성능을 확인합니다.
9. 최종 해석 단계에서는 LightGBM 모델을 중심으로 Permutation Importance와 SHAP를 분석합니다.

## 모델링 설계

### 1. 단계별 예측 구조

Open Model은 전체 발송 건을 대상으로 이메일 오픈 여부를 예측합니다.

Click-after-Open Model은 이메일을 오픈한 고객 집단 내에서 클릭 여부를 예측합니다. 따라서 이 모델은 전체 고객의 클릭 여부를 직접 예측하는 모델이 아니라, 오픈 이후 클릭 전환 패턴을 분석하기 위한 CTOR 관점의 후속 모델입니다.

### 2. 모델 비교

보고서에서는 외부 사례와의 단순 수치 비교보다, 같은 데이터와 같은 분할 조건에서 여러 모델을 비교하는 방식으로 성능을 평가합니다.

비교 대상 모델은 다음과 같습니다.

```text
Random Forest
XGBoost
LightGBM
```

모델 비교용 전처리에서는 범주형 변수에 One-Hot Encoding을 적용하고, 수치형 변수에는 결측치 대체를 적용합니다. 이를 통해 세 모델이 같은 입력 조건에서 비교되도록 구성했습니다.

### 3. 최종 XAI 모델

최종 해석은 LightGBM 모델을 중심으로 수행합니다. LightGBM은 범주형 변수와 수치형 변수가 함께 존재하는 데이터에서 비선형 관계를 포착하기 적합하고, 기존 프로젝트의 Permutation Importance 및 SHAP 분석 코드와 연결되어 있기 때문입니다.

## 평가 방식

이 프로젝트는 이메일 반응 예측 문제를 불균형 이진분류 문제로 보고, 단순 accuracy보다는 Precision, Recall, F1-score를 중심으로 모델을 평가합니다.

- Precision은 모델이 반응할 것이라고 예측한 고객 중 실제로 반응한 고객의 비율입니다.
- Recall은 실제 반응 고객 중 모델이 찾아낸 고객의 비율입니다.
- F1-score는 Precision과 Recall의 균형을 나타내는 지표입니다.

LightGBM 모델에는 `class_weight="balanced"`를 적용했습니다. 이 설정은 소수 클래스인 반응 고객을 더 잘 찾도록 도와주지만, 일반적으로 Precision은 낮아지고 Recall은 높아질 수 있습니다. 따라서 validation set에서 여러 threshold를 비교하고, Precision과 Recall의 균형이 가장 적절한 threshold를 선택했습니다.

XGBoost에는 학습 데이터의 양성/음성 비율을 이용해 `scale_pos_weight`를 적용합니다. Random Forest에는 `class_weight="balanced"`를 적용합니다.

ROC-AUC와 PR-AUC는 모델의 전반적인 순위화 성능을 확인하기 위한 보조 지표로 사용합니다. 특히 반응 고객 비율이 낮은 상황에서는 PR-AUC를 함께 확인합니다.

## 주요 출력 파일

모델링 실행 후 `A_ml_modeling_outputs` 폴더에 다음 결과가 저장됩니다.

```text
report_01_split_summary.csv       # Open / Click-after-Open split 요약
report_02_model_comparison.csv    # Random Forest / XGBoost / LightGBM 성능 비교
report_03_threshold_tuning.csv    # validation threshold 후보별 성능

model_01_split_summary.csv        # 기존 LightGBM XAI 파이프라인 split 요약
model_02_metrics.csv              # 기존 LightGBM 최종 성능
model_03_threshold_tuning.csv     # 기존 LightGBM threshold tuning 결과

xai_*                             # Permutation Importance 및 SHAP 요약 결과
```

## XAI 분석

모델 해석을 위해 다음 두 가지 방법을 사용합니다.

1. Permutation Importance  
   특정 변수 그룹을 무작위로 섞었을 때 성능이 얼마나 감소하는지를 기준으로 변수 그룹의 중요도를 확인합니다.

2. SHAP  
   개별 예측값에서 각 변수가 모델 출력값을 높이거나 낮추는 방향과 상대적 기여도를 확인합니다.

SHAP 값은 예측 확률 자체의 변화량이 아니라, 모델 출력값 기준에서 각 변수가 예측을 높이거나 낮추는 방향과 상대적 크기를 나타내는 설명 지표로 해석합니다.

## 해석상 주의점

이 프로젝트는 시간 순서를 기준으로 데이터를 분리하여 모델을 평가합니다. EDA는 전체 데이터의 반응 구조를 확인하기 위한 기술통계 분석이며, 최종 성능 평가는 test set에서 수행합니다.

balanced 설정은 소수 클래스 탐지를 강화해 Recall을 높이는 데 도움이 되지만, Precision 또는 F1-score를 단독으로 해석하면 실제 활용 성능을 과대평가할 수 있습니다. 따라서 본 프로젝트에서는 threshold tuning 결과와 PR-AUC, Precision, Recall을 함께 확인합니다.

변수 중요도와 SHAP 결과는 인과효과가 아니라, 학습된 모델이 예측에 활용한 패턴으로 해석해야 합니다.

현재 분석은 Open과 Click 단계까지를 다루며, Purchase 단계는 포함하지 않았습니다. Purchase까지 연결한 전체 퍼널 분석은 향후 연구 방향으로 남깁니다.
