import os

from common.parallel import run_parallel


def main():
    commands = []
    for sparsity_normalization in [True, False]:
        # for masking_percentage in [90]:
        for masking_percentage in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            for seed in range(1, 11):
            # for seed in [8, 9, 10]:
                commands.append(
                    'python3 -m auto_impute.trainer --sparsity_normalization={sparsity_normalization} --seed={seed} --masking_percentage={masking_percentage}'.
                    format(
                        sparsity_normalization=sparsity_normalization,
                        seed=seed,
                        masking_percentage=masking_percentage))

    run_parallel(commands, n_worker=10)


if __name__ == '__main__':
    main()
