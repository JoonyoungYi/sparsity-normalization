import os

from common.parallel import run_parallel


def main():
    for lambda_w in [
            0.00005,
            0.00006,
            0.00007,
            0.00008,
            0.00009,
            # 0.0001,
            0.0002,
            0.0003,
            0.0004,
            0.0005,
    ]:
        command = "python3 -m cfuica.trainer -seed=1 -weight-decay={lambda_w} -sparsity_normalization=y".format(
            lambda_w=lambda_w)
        commands.append(command)
    run_parallel(commands, n_worker=1)


if __name__ == '__main__':
    main()
