import os
import argparse

from common.parallel import run_parallel


def _train(args):
    # dataset = 'ujindoorloc-longitude'
    dataset = 'ujindoorloc-latitude'

    input_normalization = 'standardization'

    commands = []
    for imputation, sparsity_normalization in [
        ('mean', False),
        ('zero', False),
        ('zero', True),
    ]:
        for seed in range(1, 6):
            commands.append(
                'python3 -m uci.trainer --dataset={dataset} --sparsity_normalization={sparsity_normalization} --seed={seed} --imputation={imputation} --input_normalization={input_normalization}'.
                format(
                    dataset=dataset,
                    seed=seed,
                    imputation=imputation,
                    input_normalization=input_normalization,
                    sparsity_normalization=sparsity_normalization))
    run_parallel(commands, n_worker=15)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='uci')
    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
