import os

from common.parallel import run_parallel

# DATASET = 'blakeley'
DATASET = 'jurkat'

# DATASET = 'kolodziejczyk'
# DATASET = 'preimplantation'
# DATASET = 'quake'
# DATASET = 'usoskin'
# DATASET = 'zeisel'


def main():
    commands = []
    # for sparsity_normalization in [True]:
    #     for masking_percentage in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    #         for seed in range(1, 11):
    for masking_percentage, sparsity_normalization, seeds in [
        # (10, False, [3, 4, 5, 7, 8]),
        # (10, True, [2, 5, 6, 7]),
        # (20, False, [6, 7, 8, 9, 10]),
        # (20, True, [1, 3, 8, 9, 10]),
        # (30, False, [6, 7, 8, 9, 10]),
        # (30, True, [4, 7, 8, 9, 10]),
        (40, False, [4, 6, 7, 9, 10]),
        (40, True, [5, 6, 7, 8, 10]),
        (50, False, [6, 7, 8, 9, 10]),
        # (50, True, [5, 6, 8, 9, 10]),
            # (10, False, [1]),
            # (20, False, [1, 2, 3, 4]),
            # (30, False, [1, 2, 4]),
            # (30, True, [2, 3]),
            # (40, False, [1, 3]),
            # (40, True, [1, 4]),
            # (50, False, [1, 2, 3, 4, 5]),
            # (50, True, [1, 3]),
            # (10, False, [1]),
            # (10, True, [1, 5, 8, 9]),
            # (20, False, [1, 2, 3, 4]),
            # (20, True, [5, 6, 9]),
            # (30, False, [1, 2, 4]),
            # (30, True, [2, 3]),
            # (40, False, [1, 3]),
            # (40, True, [1, 4]),
            # (50, False, [1, 2, 3, 4, 5]),
            # (50, True, [1, 3]),
            # (60, False, [2, 3, 7, 8]),
            # (60, True, [7, 9]),
            # (70, False, [1, 3, 5, 6, 7]),
            # (70, True, [1, 2, 3, 5]),
            # (80, False, [1, 2, 5, 8, 10]),
            # (80, True, [5, 6]),
            # (90, False, [1, 2, 3, 6, 7, 8]),
            # (90, True, [2, 4, 6, 8]),
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
                # base_command += '--dataset=preimplantation --lambda_val=20 --hidden_units=4000 --learning_rate=0.001'
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

    run_parallel(commands, n_worker=15)


if __name__ == '__main__':
    main()

# lambda_val:  5(2.76): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=5   --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 10(2.58): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=10  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 15(2.55): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=15  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 20(2.52): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=20  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 30(2.53): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=30  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 40(2.58): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=40  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val:100(2.77): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=100 --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
