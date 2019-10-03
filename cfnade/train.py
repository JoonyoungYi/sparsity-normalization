import os

from common.parallel import run_parallel


def main():
    commands = []

    for seed in range(1, 6):
        for sparsity_normalization in [True]:
            # for lambda_w in [0.005, 0.010, 0.015, 0.020]:
            for lambda_w in [0.015]:
                # for lambda_w in [0.025]:
                # command_format = 'python3 -m cfnade.trainer --K_multiplier=2 --dataset_seed=1 --dataset_split=5-fold --dataset=movielens-100k --use_user_vector=False --lambda_w={lambda_w} --learning_rate=0.0001 --seed={seed} --sparsity_normalization={sparsity_normalization}'
                command_format = 'python3 -m cfnade.trainer --K_multiplier=2 --dataset_seed={seed} --dataset_split=default-10-percent --dataset=movielens-1m --use_user_vector=False --lambda_w={lambda_w} --learning_rate=0.001 --seed={seed} --sparsity_normalization={sparsity_normalization}'
                command = command_format.format(
                    lambda_w=lambda_w,
                    seed=seed,
                    sparsity_normalization=sparsity_normalization)
                commands.append(command)
    run_parallel(commands, n_worker=5)


if __name__ == '__main__':
    # python3 -m cfnade.trainer --dataset_seed=1 --dataset=movielens-100k --use_user_vector=False --lambda_w=0.003 --learning_rate=0.0002 --seed=3 --sparsity_normalization=True
    main()
