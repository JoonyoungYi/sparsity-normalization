# HVS-tensorflow

* Handling Variable Sparsity for Deep Networks에 관한 연구 실험.

## Environment

* ```python 3.5.2```
* ```tensorflow 1.12.0```

## How to initialize

* docker를 먼저 셋팅하고, ```requirements.txt``` 에 있는 package를 깔면됨.
```
sudo docker run --runtime=nvidia -it -v /home:/home tensorflow/tensorflow:1.12.0-gpu-py3 /bin/bash
cd .. && cd home/server24/HVS-tensorflow 
pip3 install -r requirements.txt
```
* ```configs/local.py.example``` 보고, ```configs/local.py``` 셋팅하기. -> 안하면 동작하다 멈출 수도 있음.


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

* MNIST 데이터셋 실험. 자세한 하이퍼파라미터는 mnist/trainer.py 파일 참조.
```
python3 -m mnist.trainer
```
