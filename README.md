# SN-tensorflow

* Official Codes for "Sparsity Normalization: Stabilizing the Expected Outputs of Deep Networks"
* Sparsity Normalization에 관한 연구 실험.

## Environment

* ```python 3.5.2```
* ```tensorflow 1.12.0```

## How to initialize

* docker를 먼저 셋팅하고, ```requirements.txt``` 에 있는 package를 깔면됨.
```
sudo docker run --runtime=nvidia -it -v /home:/home tensorflow/tensorflow:1.12.0-gpu-py3 /bin/bash
cd .. && cd home/server02/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
cd .. && cd home/server03/SN-tensorflow && export CUDA_VISIBLE_DEVICES=3
cd .. && cd home/server04/SN-tensorflow
cd .. && cd home/server05/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
cd .. && cd home/server09/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
cd .. && cd home/server12/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
cd .. && cd home/server13/SN-tensorflow
cd .. && cd home/server14/SN-tensorflow
cd .. && cd home/server17/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
cd .. && cd home/server20/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
cd .. && cd home/server22/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
cd .. && cd home/server23/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
cd .. && cd home/joonyoung/ml/SN-tensorflow && export CUDA_VISIBLE_DEVICES=0
pip3 install -r requirements.txt
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
