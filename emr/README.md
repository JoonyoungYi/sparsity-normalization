# EMR Mortality Prediction (PhysioNet 2012)

PhysioNet Challenge 2012 데이터셋을 이용한 ICU 환자 사망률 예측 실험입니다. 의료 시계열 데이터는 측정 누락이 빈번하며, 이 결측성에 SN을 적용하는 것이 핵심입니다.

## 데이터셋 준비

PhysioNet 2012 Challenge 데이터는 **직접 다운로드**가 필요합니다.

1. [https://physionet.org/content/challenge-2012/1.0.0/](https://physionet.org/content/challenge-2012/1.0.0/) 에서 데이터 사용 동의 후 다운로드
2. `set-a/`, `set-b/` 압축 해제 후 아래 경로에 배치:

```
emr/datasets/physionet2012/
├── set-a/   # 4,000개 환자 레코드 (.txt)
└── set-b/   # 4,000개 환자 레코드 (.txt)
```

3. 전처리 실행:

```bash
python3 -m emr.preprocessor
```

전처리 후 `emr/datasets/physionet2012/processed/` 에 `.npy` 파일이 생성됩니다.

## 데이터 개요

- **샘플 수**: 8,000명 ICU 환자 (set-a + set-b)
- **피처 수**: 36개 임상 변수 (HR, BP, SpO2 등)
- **시간축**: 48시간 관측, 1시간 단위 (timestep=48)
- **태스크**: 이진 분류 (병원 내 사망 여부)
- **클래스 불균형**: 사망률 약 14% → `--use_class_weight=True` 권장

## 방법 비교

| 모드 (`--mode`) | 설명 |
|-----------------|------|
| `ZI` | Zero Imputation: 결측값 → 0 |
| `MI` | Mean Imputation: 결측값 → 피처 평균 |
| `FI` | Forward Imputation: 이전 시점 값으로 대치 |
| `SN` | Sparsity Normalization (본 논문 제안 방법) |

SN 모드에서는 `--sn_imputation` 으로 전처리 방식을 추가 지정할 수 있습니다:
- `none`: 결측 타임스텝은 0으로 처리 후 SN
- `MI`: 평균 대치 후 SN
- `FI`: Forward 대치 후 SN

## 실행

```bash
pip install -r requirements.txt

# Zero Imputation (baseline)
python3 -m emr.trainer --mode=ZI

# Sparsity Normalization
python3 -m emr.trainer --mode=SN --sn_imputation=none
```

## 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `SN` | 대치 방법 |
| `--sn_imputation` | `none` | SN 전처리 방식 (mode=SN 시 유효) |

결과는 표준 출력으로 AUROC / AUPRC를 출력합니다.
