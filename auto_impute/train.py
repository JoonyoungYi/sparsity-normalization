import os

from common.parallel import run_parallel

# DATASET = 'blakeley'
# DATASET = 'jurkat'
# DATASET = 'kolodziejczyk'
DATASET = 'preimplantation'

# DATASET = 'quake'
# DATASET = 'usoskin'
# DATASET = 'zeisel'


def main():
    commands = []
    for sparsity_normalization, lambda_val in [
        (True, 20),
        (False, 2000),
    ]:
        for masking_percentage in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            for seed in range(1, 11):
                base_command = 'python3 -m auto_impute.trainer '
                if DATASET == 'blakeley':
                    base_command += '--dataset=blakeley --lambda_val=1 --hidden_units=2000 --learning_rate=0.0001'
                elif DATASET == 'jurkat':
                    # 1000이었음 원래 lambda
                    base_command += '--dataset=jurkat --lambda_val={lambda_val} --hidden_units=4000 --learning_rate=0.001'.format(
                        lambda_val=lambda_val)
                elif DATASET == 'kolodziejczyk':
                    base_command += '--dataset=kolodziejczyk --lambda_val=0 --hidden_units=8000 --learning_rate=0.0001'
                elif DATASET == 'preimplantation':
                    # base_command += '--dataset=preimplantation --lambda_val=2000 --hidden_units=4000 --learning_rate=0.001'
                    base_command += '--dataset=preimplantation --lambda_val={lambda_val} --hidden_units=4000 --learning_rate=0.0005'.format(
                        lambda_val=lambda_val)
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

    run_parallel(commands, n_worker=22)


if __name__ == '__main__':
    main()

# lambda_val:  5(2.76): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=5   --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 10(2.58): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=10  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 15(2.55): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=15  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 20(2.52): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=20  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 30(2.53): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=30  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val: 40(2.58): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=40  --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
# lambda_val:100(2.77): python3 -m auto_impute.trainer --dataset=preimplantation --lambda_val=100 --hidden_units=4000 --learning_rate=0.001 --sparsity_normalization=True --seed=1 --dataset_seed=1 --masking_percentage=10
