# Sparsity Normalization

* Official Codes for "Sparsity Normalization: Stabilizing the Expected Outputs of Deep Networks"
* Sparsity Normalization에 관한 연구 실험.

## Environment

* ```python 3.5.2```
* ```tensorflow 1.12.0```

## How to initialize

* docker 이미지 만들기
```
sh docker/build.sh
```

* docker 실행하기
```
sh docker/run.sh
cd ../home/server01/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server02/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server03/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server04/sparsity-normalization
cd ../home/server05/joonyoung_workspace/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server06/joonyoung_workspace/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/joonyoung/ml/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server09/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server10/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server12/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server13/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server14/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server17/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server19/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server20/joonyoung/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server22/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server23/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
cd ../home/server24/sparsity-normalization && export CUDA_VISIBLE_DEVICES=0
```

### How to run

* AutoRec 실험. 자세한 하이퍼파라미터는 autorec/trainer.py 파일 참조.
```
python3 -m autorec.trainer
```

* CF-NADE 실험. 자세한 하이퍼파라미터는 cfnade/trainer.py 파일 참조.
```
python3 -m cfnade.trainer
```

* UCI 데이터셋 실험. 자세한 하이퍼파라미터는 uci/trainer.py 파일 참조.
```
python3 -m uci.trainer
```

* Vision 데이터셋(MNIST, Fashion MNIST, CIFAR-10, CIFAR-100) MLP 실험.
  * 자세한 하이퍼파라미터는 vision_mlp/trainer.py 파일 참조.
```
python3 -m vision_mlp.trainer
```

* AITRICS 데이터셋 실험. 자세한 하이퍼파라미터는 aitrics/trainer.py 파일 참조.
```
python3 -m aitrics.data_preprocessing --data_path={data_path}
python3 -m aitrics.trainer --data_path={data_path}
```
