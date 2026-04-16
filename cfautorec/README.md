# CF-AutoRec with Sparsity Normalization

AutoRec(Sedhain et al., 2015) 기반 협업 필터링에 Sparsity Normalization을 적용한 실험입니다.

유저-아이템 평점 행렬은 대부분의 항목이 비어 있는 극도로 희소한 구조입니다. AutoRec은 이 희소 벡터를 오토인코더로 복원하는데, SN을 적용하면 관측 수에 따른 출력 분산을 줄여 성능이 개선됩니다.

## 데이터셋

MovieLens 데이터셋은 **첫 실행 시 자동으로 다운로드**됩니다. GroupLens 서버에서 직접 내려받습니다.

| 데이터셋 | 유저 수 | 아이템 수 | 평점 수 |
|---------|--------|---------|--------|
| MovieLens 100K | 943 | 1,682 | 100,000 |
| MovieLens 1M | 6,040 | 3,706 | 1,000,000 |
| MovieLens 10M | 69,878 | 10,677 | 10,000,054 |

데이터는 `.datasets/` 에 캐시됩니다.

## 방법 비교

| 방법 (`--imputation`) | 설명 |
|-----------------------|------|
| `zero` | 미관측 항목을 0으로 처리 |
| `mean` | 유저 평균 평점으로 대치 |
| `layer-norm` | Layer Normalization 적용 |
| `batch-norm` | Batch Normalization 적용 |

`--normalize_input=True` 로 Sparsity Normalization 활성화.

## 실행

```bash
pip install -r requirements.txt

# SN 미적용 (layer-norm baseline)
python3 -m cfautorec.trainer \
  --dataset=movielens-100k \
  --imputation=layer-norm

# SN 적용 (zero imputation + SN)
python3 -m cfautorec.trainer \
  --dataset=movielens-100k \
  --imputation=zero \
  --normalize_input=True \
  --K_denominator=500

# 전체 실험 (병렬)
python3 -m cfautorec.train
```

## 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `movielens-100k` | 데이터셋 |
| `--imputation` | `layer-norm` | 대치 방법 |
| `--normalize_input` | `False` | SN 적용 여부 |
| `--K_denominator` | `500.0` | SN의 K 값 (평균 관측 평점 수) |
| `--n_hidden_unit` | `500` | 은닉층 유닛 수 |
| `--lambda_w` | `500.0` | L2 정규화 계수 |
| `--use_user_vector` | `True` | True=user 기준, False=item 기준 |
| `--activation_fn` | `sigmoid` | 활성화 함수 |

결과는 `.results/autorec/{dataset}/` 에 저장됩니다.
