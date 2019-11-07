# CF-UIcA

This repository contains the implementation for the paper:

**Chao Du, Chongxuan Li, Yin Zheng, Jun Zhu, Bo Zhang. [Collaborative Filtering with User-Item Co-Autoregressive Models.](https://arxiv.org/abs/1612.07146)** In AAAI 2018.

## Datasets

To download and preprocess the Movielens 1M dataset:

```
$ cd data/
$ ./download.sh
$ python preprocess.py
```

## Usage

To run rating prediction experiments:

```
$ python -m cfuica.trainer
```
