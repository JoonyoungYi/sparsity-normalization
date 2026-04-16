# Sparsity Normalization

**"Sparsity Normalization: Stabilizing the Expected Outputs of Deep Networks"** 공식 구현 코드

> arXiv: https://arxiv.org/abs/1906.00150

---

## 개요

희소 입력(sparse input)이 있을 때 신경망의 출력이 불안정해지는 **Variable Sparsity Problem (VSP)** 을 분석하고, 이를 해결하는 **Sparsity Normalization (SN)** 기법을 제안합니다.

Sparsity Normalization은 각 샘플의 관찰된 피처 수 $c_i$로 출력을 정규화합니다:

$$\hat{y}_i = f\left(\mathbf{x}_i \cdot W\right) \cdot \frac{K'}{c_i}$$

여기서 $K'$는 학습 데이터의 평균 관찰 피처 수입니다.

---

## 실험 모듈

| 모듈 | 설명 | 데이터셋 |
|------|------|---------|
| `uci/` | UCI 회귀 실험 (결측값 삽입) | Boston, Diabetes, California 등 (sklearn) |
| `cfautorec/` | AutoRec 협업 필터링 | MovieLens 100K / 1M / 10M |
| `cfnade/` | CF-NADE 협업 필터링 | MovieLens 100K / 1M |
| `emr/` | 의료 시계열 사망률 예측 (LSTM) | PhysioNet 2012 Challenge |

---

## 환경 설정

### Docker (권장)

```bash
sh docker/build.sh
sh docker/run.sh
```

### 직접 설치

```bash
pip install tensorflow==1.12.0 torch numpy scikit-learn tqdm
```

---

## 실행 방법

### UCI 회귀 실험

```bash
python3 -m uci.trainer \
  --dataset=boston \
  --sparsity_normalization=True \
  --imputation=zero \
  --missing_rate=0.5
```

전체 실험 (병렬):

```bash
python3 -m uci.train
```

### AutoRec 협업 필터링

```bash
python3 -m cfautorec.trainer \
  --dataset=movielens-100k \
  --normalize_input=True \
  --imputation=layer-norm
```

### CF-NADE 협업 필터링

```bash
python3 -m cfnade.trainer
```

### PhysioNet 2012 의료 시계열

데이터셋은 [PhysioNet](https://physionet.org/content/challenge-2012/1.0.0/) 에서 직접 다운로드 후 `emr/datasets/physionet2012/` 에 압축 해제하세요.

전처리 후 실행:

```bash
python3 -m emr.preprocessor
python3 -m emr.trainer --mode=SN --sn_imputation=none
```

---

## 인용

```bibtex
@article{yi2019sparsity,
  title={Sparsity Normalization: Stabilizing the Expected Outputs of Deep Networks},
  author={Yi, Joonyoung},
  journal={arXiv preprint arXiv:1906.00150},
  year={2019}
}
```
