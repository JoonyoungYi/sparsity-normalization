import os

from common.parallel import run_parallel

DATASET = 'blakeley'

# DATASET = 'jurkat'
# DATASET = 'kolodziejczyk'
# DATASET = 'preimplantation'
# DATASET = 'quake'
# DATASET = 'usoskin'
# DATASET = 'zeisel'


def main():
    commands = []
    # for sparsity_normalization in [True, False]:
    #     for masking_percentage in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    #         for seed in range(1, 11):

    for masking_percentage, sparsity_normalization, seeds in [
        (10, False, [1, 3]),
        (10, True, [1, 5, 8, 9]),
        (20, False, [1, 2, 5, 6, 7, 9]),
        (20, True, [5, 6, 9]),
        (30, False, [1, 2, 6, 7, 8]),
        (30, True, [2, 5, 6]),
        (40, False, [1, 6, 7, 8]),
        (40, True, [5, 7, 8, 9, 10]),
        (50, False, [5, 6, 7]),
        (50, True, [2, 5, 7]),
        (60, False, [2, 3, 7, 8]),
        (60, True, [7, 9]),
        (70, False, [1, 3, 5, 6, 7]),
        (70, True, [1, 2, 3, 5]),
        (80, False, [1, 2, 5, 8, 10]),
        (80, True, [5, 6]),
        (90, False, [1, 2, 3, 6, 7, 8]),
        (90, True, [2, 4, 6, 8]),
    ]:
        for seed in seeds:
            base_command = 'python3 -m auto_impute.trainer '
            if DATASET == 'blakeley':
                base_command += '--dataset=blakeley --lambda_val=1 --hidden_units=2000 --learning_rate=0.0001'
            elif DATASET == 'jurkat':
                base_command += '--dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.001'
            elif DATASET == 'kolodziejczyk':
                base_command += '--dataset=kolodziejczyk --lambda_val=0 --hidden_units=8000 --learning_rate=0.0001'
            elif DATASET == 'preimplantation':
                base_command += '--dataset=preimplantation --lambda_val=2000 --hidden_units=4000 --learning_rate=0.001'
            elif DATASET == 'quake':
                base_command += '--dataset=quake --lambda_val=2000 --hidden_units=1500 --learning_rate=0.0001'
            elif DATASET == 'usoskin':
                base_command += '--dataset=usoskin --lambda_val=2000 --hidden_units=3000 --learning_rate=0.0001'
            elif DATASET == 'zeisel':
                base_command += '--dataset=zeisel --lambda_val=1000 --hidden_units=1500 --learning_rate=0.0001'
            else:
                raise NotImplementedError()

            commands.append(
                base_command +
                ' --sparsity_normalization={sparsity_normalization} --seed={seed} --dataset_seed={seed} --masking_percentage={masking_percentage}'.
                format(
                    sparsity_normalization=sparsity_normalization,
                    seed=seed,
                    masking_percentage=masking_percentage))

    run_parallel(commands, n_worker=10)


if __name__ == '__main__':
    main()
