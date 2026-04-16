import argparse

from common.parallel import run_parallel
from .trainer import DATASETS


def _train(args):
    commands = []
    for dataset in DATASETS:
        for imputation, sparsity_normalization in [
            ('mean', False),
            ('zero', False),
            ('zero', True),
        ]:
            for seed in range(1, 6):
                commands.append(
                    'python3 -m uci.trainer'
                    ' --dataset={dataset}'
                    ' --sparsity_normalization={sn}'
                    ' --seed={seed}'
                    ' --imputation={imputation}'
                    ' --input_normalization=min-max'.format(
                        dataset=dataset,
                        seed=seed,
                        imputation=imputation,
                        sn=sparsity_normalization))
    run_parallel(commands, n_worker=15)


def main():
    parser = argparse.ArgumentParser(description='UCI 전체 실험 병렬 실행')
    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
