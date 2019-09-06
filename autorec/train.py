import os

from common.parallel import run_parallel


def _make_command(lambda_w, normalize_input, use_user_vector):
    return 'python3 -m autorec.trainer --lambda_w={lambda_w} --normalize_input={normalize_input} --K_denominator=500 --use_user_vector={use_user_vector} --activation_fn=relu'.format(
        lambda_w=lambda_w,
        normalize_input=normalize_input,
        use_user_vector=use_user_vector)


def _get_commands():
    commands = []
    for i in range(0, 5):
        for j in range(1, 10):
            lambda_w = (10**i) * j
            for normalize_input in [True, False]:
                for use_user_vector in [True, False]:
                # for use_user_vector in [True]:
                    command = _make_command(
                        lambda_w=lambda_w,
                        normalize_input=normalize_input,
                        use_user_vector=use_user_vector)
                    commands.append(command)
    return commands


def main():
    commands = [
        # "python3 -m autorec.trainer --lambda_w=300 --save_weights=True --normalize_input=False --use_user_vector=True",
        # "python3 -m autorec.trainer --lambda_w=600 --save_weights=True --normalize_input=True --use_user_vector=True",
        # "python3 -m autorec.trainer --lambda_w=300 --save_weights=True --normalize_input=False --use_user_vector=False",
        # "python3 -m autorec.trainer --lambda_w=600 --save_weights=True --normalize_input=True --use_user_vector=False"
        "python3 -m autorec.trainer --lambda_w=7000 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=identity",
        "python3 -m autorec.trainer --lambda_w=2000 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=identity",
        "python3 -m autorec.trainer --lambda_w=8000 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=identity",
        "python3 -m autorec.trainer --lambda_w=2000 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=identity"
        "python3 -m autorec.trainer --lambda_w=8000 --save_weights=True --normalize_input=True --use_user_vector=False --activation_fn=relu",
        "python3 -m autorec.trainer --lambda_w=2000 --save_weights=True --normalize_input=False --use_user_vector=False --activation_fn=relu",
        "python3 -m autorec.trainer --lambda_w=8000 --save_weights=True --normalize_input=True --use_user_vector=True --activation_fn=relu",
        "python3 -m autorec.trainer --lambda_w=2000 --save_weights=True --normalize_input=False --use_user_vector=True --activation_fn=relu"
    ]
    # commands = _get_commands()
    run_parallel(commands, n_worker=10)


if __name__ == '__main__':
    main()
