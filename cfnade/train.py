import os

from common.parallel import run_parallel


def main():
    commands = []

    learning_rate = 0.001
    for seed in range(1, 6):
        for sparsity_normalization in [True, False]:
            for lambda_w in [0.005, 0.01, 0.015, 0.02]:
            # for lambda_w in [0.005]:
                command = 'python3 -m cfnade.trainer --dataset=movielens-1m --use_user_vector=True --lambda_w={lambda_w} --learning_rate={learning_rate} --seed={seed} --sparsity_normalization={sparsity_normalization}'.format(
                    learning_rate=learning_rate,
                    lambda_w=lambda_w,
                    seed=seed,
                    sparsity_normalization=sparsity_normalization)
                commands.append(command)
    run_parallel(commands, n_worker=5)


if __name__ == '__main__':
    main()
