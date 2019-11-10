import os


def main():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-seed',
    #     type=int,
    #     default=12345,
    #     help="numpy random seed for reproducibility")
    # args = parser.parse_args()

    commands = []
    for lambda_w in [
            # 0.00005,
            # 0.00006,
            # 0.00007,
            # 0.00008,
            # 0.00009,
            0.0001,
            # 0.0002,
            # 0.0003,
            # 0.0004,
            # 0.0005,
    ]:
        for seed in [12345, 1, 2, 3, 4]:
            command = "python trainer.py -seed={seed} -weight_decay={lambda_w} -sparsity_normalization=y".format(
                lambda_w=lambda_w, seed=seed)
            commands.append(command)

    for command in commands:
        os.system(command)


if __name__ == '__main__':
    main()
