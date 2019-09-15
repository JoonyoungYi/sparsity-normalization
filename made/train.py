import os

from common.parallel import run_parallel


def main():
    commands = []
    for num_masks in range(1, 128):
        for seed in range(1, 6):
            for sparsity_normalization in [True, False]:
                command = 'python3 -m made.trainer --num_masks={num_masks} --samples={num_masks} --learning_rate=5e-3 --seed={seed} --sparsity_normalization={sparsity_normalization}'.format(
                    num_masks=num_masks,
                    seed=seed,
                    sparsity_normalization=sparsity_normalization)
                commands.append(command)
        run_parallel(commands, n_worker=10)


if __name__ == '__main__':
    main()
