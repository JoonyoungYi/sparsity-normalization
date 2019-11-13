import os

from common.parallel import run_parallel


def _make_command(lambda_w, normalize_input, use_user_vector, activation_fn):
    return 'python3 -m autorec.trainer --lambda_w={lambda_w} --normalize_input={normalize_input} --K_denominator=500 --use_user_vector={use_user_vector} --activation_fn={activation_fn} --clipping=True'.format(
        lambda_w=lambda_w,
        normalize_input=normalize_input,
        activation_fn=activation_fn,
        use_user_vector=use_user_vector)


def _get_commands():
    commands = []
    # for normalize_input in [True, False]:
    for activation_fn in [
            # 'identity', 'relu', 'relu6', 'sigmoid', 'tanh', 'leaky_relu'
            'softplus',
    ]:
        for normalize_input in [True]:
            # for use_user_vector in [True]:
            for use_user_vector in [True, False]:
                command = _make_command(
                    lambda_w=0.0,
                    normalize_input=normalize_input,
                    use_user_vector=use_user_vector,
                    activation_fn=activation_fn)
                commands.append(command)

                for i in range(0, 5):
                    for j in range(1, 10):
                        lambda_w = (10**i) * j
                        command = _make_command(
                            lambda_w=lambda_w,
                            normalize_input=normalize_input,
                            use_user_vector=use_user_vector,
                            activation_fn=activation_fn)
                        commands.append(command)
    return commands


def main():
    # commands = [
    #     # # identity
    #     # "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=identity --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=7000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=identity --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=700 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=identity --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=identity --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=8000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=identity --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=800 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=identity --clipping=True",
    #     # # leaky_relu
    #     # "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=leaky_relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=7000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=leaky_relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=700 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=leaky_relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=leaky_relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=8000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=leaky_relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=900 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=leaky_relu --clipping=True",
    #     # # relu
    #     # "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=6000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=700 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=8000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=relu --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=900 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=relu --clipping=True",
    #     # # relu6
    #     # "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=relu6 --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=7000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=relu6 --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=800 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=relu6 --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=relu6 --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=7000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=relu6 --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=800 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=relu6 --clipping=True",
    #     # # sigmoid
    #     # "python3 -m autorec.trainer --lambda_w=300 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=sigmoid --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=600 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=sigmoid --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=100 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=sigmoid --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=300 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=sigmoid --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=500 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=sigmoid --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=200 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=sigmoid --clipping=True",
    #     # # tanh
    #     # "python3 -m autorec.trainer --lambda_w=900 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=tanh --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=3000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=tanh --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=500 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=tanh --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=800 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=tanh --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=1000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=tanh --clipping=True",
    #     # "python3 -m autorec.trainer --lambda_w=500 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=tanh --clipping=True",
    #     # elu
    #     "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=elu --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=7000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=elu --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=700 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=elu --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=elu --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=800 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=elu --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=1000 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=elu --clipping=True",
    #     # softplus
    #     "python3 -m autorec.trainer --lambda_w=2000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=softplus --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=6000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=softplus --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=600 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=softplus --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=1000 --K_denominator=500 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=softplus --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=7000 --K_denominator=500 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=softplus --clipping=True",
    #     "python3 -m autorec.trainer --lambda_w=700 --K_denominator=52.44 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=softplus --clipping=True",
    # ]
    # commands = _get_commands()

    # commands = []

    # imputation = 'mean'
    # imputation = 'median'
    # imputation = 'knn'
    # imputation = 'mice'
    # imputation = 'soft-impute'

    # use_user_vector = True
    # use_user_vector = False

    # for lambda_w in [
    #         0, 1, 20, 30, 40, 50, 70, 80, 90, 100, 200, 600, 700, 800, 900
    # ]:
    #     for seed in [1]:
    #         command = "python3 -m cfautorec.trainer --imputation={imputation} --lambda_w={lambda_w} --use_user_vector={use_user_vector} --seed={seed}".format(
    #             lambda_w=lambda_w,
    #             use_user_vector=use_user_vector,
    #             imputation=imputation,
    #             seed=seed)
    #         commands.append(command)

    # dataset = 'movielens-100k'
    dataset = 'movielens-1m'

    # 0,
    # 1,
    # 10,
    # 100,
    # 1000,
    # 10000,
    # 100000,
    # 1000000,
    # 10000000,
    # 100000000,
    commands = []
    for lambda_w in [
            70,
            400,
            700,
            4000,
            7000,
    ]:
        # for imputation in ['batch-norm']:
        # for imputation in ['layer-norm']:
        # for imputation in ['soft-impute', 'gain']:
        # for imputation in ['soft-impute']:
        # for imputation in ['dropout']:
        # for imputation in ['gain']:
        for imputation in ['gmmc']:
            # for imputation in ['batch-norm', 'layer-norm']:
            # for imputation in ['knn', 'mean', 'median']:
            # for imputation in ['gmmc', 'soft-impute']:

            for use_user_vector in [True]:
            # for use_user_vector in [False]:
                # for use_user_vector in [True, False]:

                # for seed in [2, 3, 4, 5]:
                # for seed in [6, 7, 8, 9, 10]:
                for seed in [1]:
                    if imputation != 'gmmc':
                        command = "python3 -m cfautorec.trainer --dataset={dataset} --imputation={imputation} --lambda_w={lambda_w} --use_user_vector={use_user_vector} --seed={seed}".format(
                            lambda_w=lambda_w,
                            dataset=dataset,
                            use_user_vector=use_user_vector,
                            imputation=imputation,
                            seed=seed)
                    else:
                        command = "python3 -m cfautorec.gmmc_trainer --dataset={dataset} --lambda_w={lambda_w} --use_user_vector={use_user_vector} --seed={seed}".format(
                            lambda_w=lambda_w,
                            dataset=dataset,
                            use_user_vector=use_user_vector,
                            seed=seed)

                    if imputation in ['layer-norm', 'batch-norm']:
                        command = command + ' --epoch_early_stop=10000'

                    commands.append(command)

    run_parallel(commands, n_worker=1)


if __name__ == '__main__':
    main()
