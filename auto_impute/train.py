import os

from common.parallel import run_parallel

# DATASET = 'blakeley'
# DATASET = 'jurkat'
DATASET = 'kolodziejczyk'
# DATASET = 'preimplantation'
# DATASET = 'quake'
# DATASET = 'usoskin'
# DATASET = 'zeisel'


def main():
    commands = []
    sparsity_normalization = False
    for masking_percentage in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        for seed in range(1, 11):
            base_command = 'python3 -m auto_impute.trainer '
            if DATASET == 'blakeley':
                threshold = 1e-4
                base_command += '--dataset=blakeley --lambda_val=1 --hidden_units=2000 --learning_rate=0.0001 --threshold={threshold}'.format(
                    threshold=threshold)
            elif DATASET == 'jurkat':
                lambda_val = 1000
                threshold = 1e-3
                base_command += '--dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.001 --threshold={threshold}'.format(
                    lambda_val=lambda_val, threshold=threshold)
            elif DATASET == 'kolodziejczyk':
                threshold = 1e-4
                base_command += '--dataset=kolodziejczyk --lambda_val=0 --hidden_units=8000 --learning_rate=0.0001 --threshold={threshold}'.format(
                    threshold=threshold)
            elif DATASET == 'preimplantation':
                lambda_val = 2000
                threshold = 1e-3
                base_command += '--dataset=preimplantation --lambda_val={lambda_val} --hidden_units=4000 --learning_rate=0.001 --threshold={threshold}'.format(
                    lambda_val=lambda_val, threshold=threshold)
            elif DATASET == 'quake':
                base_command += '--dataset=quake --lambda_val=2000 --hidden_units=1500 --learning_rate=0.0001 --threshold={threshold}'.format(
                    threshold=threshold)
            elif DATASET == 'usoskin':
                base_command += '--dataset=usoskin --lambda_val=2000 --hidden_units=3000 --learning_rate=0.0001 --threshold={threshold}'.format(
                    threshold=threshold)
            elif DATASET == 'zeisel':
                base_command += '--dataset=zeisel --lambda_val=1000 --hidden_units=1500 --learning_rate=0.0001 --threshold={threshold}'.format(
                    threshold=threshold)
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

# [1]: python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=2.4 --hidden_units=4000 --learning_rate=0.0005 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=90
# [9]: python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=2.4 --hidden_units=4000 --learning_rate=0.0001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=90

# [2]: python3 -m auto_impute.trainer --dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=70
# [3]: python3 -m auto_impute.trainer --dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.0001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=70
# [5]: python3 -m auto_impute.trainer --dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.0002 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=70
# [6]: python3 -m auto_impute.trainer --dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.0005 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=70

# [8]: python3 -m auto_impute.trainer --dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=False --seed=1 --dataset_seed=1 --masking_percentage=70 --threshold=1e-3

# [0]: python3 -m auto_impute.trainer --dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.0005 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=80
# [4]: python3 -m auto_impute.trainer --dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.0005 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=60
# [7]: python3 -m auto_impute.trainer --dataset=jurkat --lambda_val=1000 --hidden_units=4000 --learning_rate=0.0005 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=90
