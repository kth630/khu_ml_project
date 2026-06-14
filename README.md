# Trigger Email Funnel Analysis

이 저장소는 이메일 마케팅 데이터를 바탕으로 Trigger Email의 반응 흐름을 분석한 머신러닝 프로젝트입니다.

주요 내용은 이메일 발송 이후 고객이 메일을 열었는지(`open`), 그리고 오픈 후 클릭까지 이어졌는지(`click_after_open`)를 예측하고 해석하는 코드입니다.

최종 제출용 분석 코드는 `email_ml/` 폴더에 정리되어 있으며, 전처리, EDA, 모델링, 성능 비교, XAI 결과 생성 과정을 포함합니다.

```text
email_ml/
├── 01_data_preprocessing.py
├── 02_exploratory_analysis.py
├── 03_modeling_and_evaluation.py
└── outputs/
```

원본 데이터와 학습된 모델 파일은 용량 문제로 GitHub 제출 대상에서 제외했습니다.
