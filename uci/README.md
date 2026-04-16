# UCI Regression Experiments

UCI 공개 데이터셋에 결측값을 인위적으로 삽입한 뒤, 다양한 대치(imputation) 방식과 Sparsity Normalization(SN)의 성능을 비교합니다.

## 방법 비교

| 방법 | 설명 |
|------|------|
| Zero Imputation (ZI) | 결측값을 0으로 대치 |
| Mean Imputation (MI) | 학습 데이터의 평균으로 대치 |
| Zero + SN | 0으로 대치 후 SN 적용 |

**Sparsity Normalization**은 각 샘플의 관찰된 피처 수 $c_i$로 출력을 보정합니다:

$$\tilde{x}_i = x_i \cdot \frac{K'}{c_i}$$

여기서 $K' = \mathbb{E}[c_i]$ (학습셋 평균 관찰 수)

## 데이터셋

scikit-learn에서 자동으로 로드됩니다. 별도 다운로드 불필요.

| 데이터셋 | 샘플 수 | 피처 수 | 태스크 |
|---------|--------|--------|------|
| Boston Housing | 506 | 13 | 회귀 |
| Diabetes | 442 | 10 | 회귀 |
| California Housing | 20,640 | 8 | 회귀 |
| Wine Quality | 178 | 13 | 분류 |
| Breast Cancer | 569 | 30 | 분류 |

## 실행

```bash
pip install -r requirements.txt

# 단일 실험
python3 -m uci.trainer \
  --dataset=boston \
  --sparsity_normalization=True \
  --imputation=zero \
  --missing_rate=0.5

# 전체 실험 (병렬)
python3 -m uci.train
```

## 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `boston` | 데이터셋 선택 |
| `--missing_rate` | `0.5` | 결측값 삽입 비율 |
| `--sparsity_normalization` | `False` | SN 적용 여부 |
| `--imputation` | `zero` | 결측값 대치 방법 (`zero` / `mean`) |
| `--input_normalization` | `min-max` | 입력 정규화 방법 |
| `--seed` | `1` | 랜덤 시드 |

결과는 `.results/uci/{dataset}/` 에 저장됩니다.
