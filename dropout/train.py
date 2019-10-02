import os

from .trainer import parse_arguments
from common.parallel import run_parallel


def _make_command(dropout_mode, dropout_rate, learning_rate, optimizer,
                  activation_fn, n_hidden_layer, n_hidden_unit, n_batch):
    return 'python3 -m dropout.trainer --dropout_mode={dropout_mode} --dropout_rate={dropout_rate} --optimizer={optimizer} --learning_rate={learning_rate} --n_hidden_unit={n_hidden_unit} --n_hidden_layer={n_hidden_layer} --activation_fn={activation_fn} --n_batch={n_batch}'.format(
        dropout_mode=dropout_mode,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        optimizer=optimizer,
        n_hidden_layer=n_hidden_layer,
        n_hidden_unit=n_hidden_unit,
        n_batch=n_batch,
        activation_fn=activation_fn)


def _get_commands(args):
    activation_fn = args.activation_fn
    optimizer = args.optimizer
    n_hidden_unit = args.n_hidden_unit
    n_hidden_layer = args.n_hidden_layer
    learning_rate = args.learning_rate
    n_batch = args.n_batch

    commands = [
        # baseline
        _make_command(
            dropout_mode='default',
            dropout_rate=0.0,
            optimizer=optimizer,
            learning_rate=learning_rate,
            activation_fn=activation_fn,
            n_hidden_layer=n_hidden_layer,
            n_hidden_unit=n_hidden_unit,
            n_batch=n_batch)
    ]
    for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for dropout_mode in ['default', 'sparsity-normalization']:
            command = _make_command(
                dropout_mode=dropout_mode,
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                learning_rate=learning_rate,
                activation_fn=activation_fn,
                n_hidden_layer=n_hidden_layer,
                n_hidden_unit=n_hidden_unit,
                n_batch=n_batch)
            commands.append(command)
    return commands


def _train(args):
    # commands = _get_commands(args)
    commands = []
    for dropout_mode in ['default', 'sparsity-normalization']:
        for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for seed in [1, 2, 3, 4, 5]:
                commands.append(
                    'python3 -m dropout.trainer --dropout_mode={dropout_mode} --seed={seed} --dataset_seed=1 --dropout_rate={dropout_rate} --dataset=wine --learning_rate=1e-4'.
                    format(
                        dropout_mode=dropout_mode,
                        dropout_rate=dropout_rate,
                        seed=seed))
    run_parallel(commands, n_worker=10)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='dropout')
    parse_arguments(parser)
    args = parser.parse_args()

    _train(args)


# RQ4. batch size에 어떻게 영향을 받는가?
# [torch-12] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=8
# [torch-9] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=16
# [torch-8] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=32
# [torch-8] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=64
# [torch-13] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=128
# python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=256
# [torch-14] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=512
# python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=0

if __name__ == '__main__':
    main()
