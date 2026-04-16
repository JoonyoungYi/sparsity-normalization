# CF-NADE with Sparsity Normalization

CF-NADE(Zheng et al., 2016) 기반 협업 필터링에 Sparsity Normalization을 적용한 실험입니다.

NADE(Neural Autoregressive Distribution Estimator)는 평점 분포를 자기회귀 방식으로 모델링합니다. SN을 적용하면 관측 수가 다른 유저들 사이에서 출력 기댓값을 안정적으로 유지합니다.

## 데이터셋

MovieLens 데이터셋은 **첫 실행 시 자동으로 다운로드**됩니다.

| 데이터셋 | 유저 수 | 아이템 수 | 평점 수 |
|---------|--------|---------|--------|
| MovieLens 100K | 943 | 1,682 | 100,000 |
| MovieLens 1M | 6,040 | 3,706 | 1,000,000 |

데이터는 `.datasets/` 에 캐시됩니다.

## 데이터 분할 방식

| `--dataset_split` | 설명 |
|-------------------|------|
| `default-10-percent` | 전체의 8.82% / 1.8% / 10%를 train/valid/test로 분할 |
| `5-fold` | MovieLens 100K의 공식 5-fold 분할 사용 (seed 1~5) |

## 실행

```bash
pip install -r requirements.txt

# SN 미적용 baseline
python3 -m cfnade.trainer \
  --dataset=movielens-1m \
  --sparsity_normalization=False

# SN 적용
python3 -m cfnade.trainer \
  --dataset=movielens-1m \
  --sparsity_normalization=True \
  --K_multiplier=2.0

# 전체 실험 (병렬)
python3 -m cfnade.train
```

## 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `movielens-1m` | 데이터셋 |
| `--dataset_split` | `default-10-percent` | 데이터 분할 방식 |
| `--sparsity_normalization` | `False` | SN 적용 여부 |
| `--K_multiplier` | `2.0` | SN의 K 배율 |
| `--n_hidden_unit` | `500` | 은닉층 유닛 수 |
| `--lambda_w` | `0.015` | L2 정규화 계수 |
| `--parameter_sharing` | `True` | 인코더/디코더 가중치 공유 |
| `--use_ordinal_cost` | `True` | 순서형 손실 함수 사용 |
| `--use_user_vector` | `False` | True=user 기준, False=item 기준 |

결과는 `.results/cfnade/{dataset}/` 에 저장됩니다.
