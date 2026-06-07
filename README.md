# Email Marketing ML Project

이 저장소는 이메일 마케팅 데이터를 이용해 이메일 오픈 여부와 오픈 이후 클릭 여부를 예측하는 머신러닝 프로젝트 코드입니다.

## 파일 구성

```text
A_ml_preprocessing.py  # 원본 데이터 전처리 및 모델링 데이터셋 생성
A_ml_eda.py            # EDA 표와 시각화 생성
A_ml_modeling.py       # 모델 학습, 평가, XAI 분석
requirements.txt       # 실행에 필요한 패키지 목록
```

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
python A_ml_preprocessing.py
python A_ml_eda.py
python A_ml_modeling.py
```

## 분석 흐름

1. trigger 캠페인과 email 채널 메시지를 필터링합니다.
2. 고객별 과거 이메일 수신, 오픈, 클릭 이력을 생성합니다.
3. 발송 시각, 주제, 이메일 제공자, 과거 반응 이력 변수를 사용해 모델을 학습합니다.
4. 시간 순서를 기준으로 train, validation, test set을 분리합니다.
5. LightGBM 모델로 오픈 여부와 오픈 이후 클릭 여부를 예측합니다.
6. PR-AUC, ROC-AUC, Precision@K, Recall@K, Lift@K 등을 계산합니다.
7. Permutation Importance와 SHAP를 이용해 모델 예측 패턴을 해석합니다.

## 모델 설명

Open Model은 전체 발송 건을 대상으로 이메일 오픈 여부를 예측합니다.

Click-after-Open Model은 이메일을 오픈한 고객 집단 내에서 클릭 여부를 예측합니다. 따라서 이 모델은 전체 고객의 클릭 여부를 직접 예측하는 모델이 아니라, 오픈 이후 클릭 전환 패턴을 분석하기 위한 후속 모델입니다.

## 해석상 주의점

이 프로젝트는 시간 순서를 기준으로 데이터를 분리하여 모델을 평가합니다. 불균형 분류 문제이므로 accuracy보다는 PR-AUC와 Top-K 기반 지표를 중심으로 해석합니다.

SHAP 값은 예측 확률 자체의 변화량이 아니라, 모델 출력값 기준에서 각 변수가 예측을 높이거나 낮추는 방향과 상대적 크기를 나타내는 설명 지표로 해석합니다.

동일 고객에게 동일 발송 시각에 여러 메시지가 존재하는 경우, 동일 시각 내 메시지 순서에 따른 미세한 이력 반영 가능성이 있을 수 있습니다. 따라서 이 부분은 분석의 한계로 둡니다.

변수 중요도와 SHAP 결과는 인과효과가 아니라, 학습된 모델이 예측에 활용한 패턴으로 해석해야 합니다.
