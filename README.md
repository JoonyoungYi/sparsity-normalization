# Sparsity Normalization

**"Sparsity Normalization: Stabilizing the Expected Outputs of Deep Networks"** 공식 구현 코드

> arXiv: https://arxiv.org/abs/1906.00150

---

## 개요

희소 입력(sparse input)이 있을 때 신경망의 출력이 불안정해지는 **Variable Sparsity Problem (VSP)** 을 분석하고, 이를 해결하는 **Sparsity Normalization (SN)** 을 제안합니다.

SN은 각 샘플에서 관찰된 피처 수 $c_i$로 출력을 정규화합니다:

$$\hat{h} = f(xW) \cdot \frac{K'}{c_i}$$

$K' = \mathbb{E}[c_i]$ 는 학습 데이터 평균 관찰 수로, SN은 관측 수에 무관하게 기댓값을 일정하게 유지합니다.

---

## 실험 모듈

| 모듈 | 프레임워크 | 설명 | 데이터셋 |
|------|-----------|------|---------|
| [`uci/`](uci/README.md) | PyTorch | UCI 회귀 (결측값 삽입 실험) | sklearn 자동 로드 |
| [`cfautorec/`](cfautorec/README.md) | TensorFlow 1.x | AutoRec 협업 필터링 | MovieLens (자동 다운로드) |
| [`cfnade/`](cfnade/README.md) | TensorFlow 1.x | CF-NADE 협업 필터링 | MovieLens (자동 다운로드) |
| [`emr/`](emr/README.md) | PyTorch | ICU 사망률 예측 (LSTM) | PhysioNet 2012 (**직접 다운로드 필요**) |

---

## 설치

각 모듈은 독립적인 환경에서 실행 가능합니다. 사용할 모듈 폴더의 `requirements.txt` 를 설치하세요.

```bash
# PyTorch 기반 모듈 (uci, emr)
pip install torch numpy scikit-learn tqdm

# TensorFlow 기반 모듈 (cfautorec, cfnade)
pip install tensorflow==1.12.0 numpy scipy requests tqdm
```

---

## 빠른 시작

```bash
# UCI 회귀 실험
python3 -m uci.trainer --dataset=boston --sparsity_normalization=True

# AutoRec 협업 필터링
python3 -m cfautorec.trainer --dataset=movielens-100k --normalize_input=True

# CF-NADE 협업 필터링
python3 -m cfnade.trainer --dataset=movielens-1m --sparsity_normalization=True

# PhysioNet 2012 사망률 예측 (데이터 준비 후)
python3 -m emr.preprocessor
python3 -m emr.trainer --mode=SN
```

자세한 실행 방법은 각 모듈의 README를 참고하세요.

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
