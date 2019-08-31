import os

from .trainer import parse_arguments


def _run(dropout_mode, dropout_rate, learning_rate, optimizer, activation_fn,
         n_hidden_layer, n_hidden_unit, n_batch):
    os.system(
        'python3 -m dropout.trainer --dropout_mode={dropout_mode} --dropout_rate={dropout_rate} --optimizer={optimizer} --learning_rate={learning_rate} --n_hidden_unit={n_hidden_unit} --n_hidden_layer={n_hidden_layer} --activation_fn={activation_fn} --n_batch={n_batch}'.
        format(
            dropout_mode=dropout_mode,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            optimizer=optimizer,
            n_hidden_layer=n_hidden_layer,
            n_hidden_unit=n_hidden_unit,
            n_batch=n_batch,
            activation_fn=activation_fn))


def _train(args):
    activation_fn = args.activation_fn
    optimizer = args.optimizer
    n_hidden_unit = args.n_hidden_unit
    n_hidden_layer = args.n_hidden_layer
    learning_rate = args.learning_rate
    n_batch = args.n_batch

    # baseline learning
    _run(
        dropout_mode='default',
        dropout_rate=0.0,
        optimizer=optimizer,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
        n_hidden_layer=n_hidden_layer,
        n_hidden_unit=n_hidden_unit,
        n_batch=n_batch)

    for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for dropout_mode in ['default', 'sparsity-normalization']:
            _run(
                dropout_mode=dropout_mode,
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                learning_rate=learning_rate,
                activation_fn=activation_fn,
                n_hidden_layer=n_hidden_layer,
                n_hidden_unit=n_hidden_unit,
                n_batch=n_batch)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='dropout')
    parse_arguments(parser)
    args = parser.parse_args()

    _train(args)


# RQ1. Activation Function의 변화에 따른 성능의 변화
# [torch-0] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3
# [torch-0] python3 -m dropout.train --optimizer=adam --activation_fn=leaky_relu --n_hidden_layer=2 --learning_rate=1e-3
# [torch-1] python3 -m dropout.train --optimizer=adam --activation_fn=sigmoid --n_hidden_layer=2 --learning_rate=1e-3
# [torch-1] python3 -m dropout.train --optimizer=adam --activation_fn=tanh --n_hidden_layer=2 --learning_rate=1e-3
# [torch-2] python3 -m dropout.train --optimizer=adam --activation_fn=elu --n_hidden_layer=2 --learning_rate=1e-3

# RQ2. optimizer와 learning rate에 따라 어떻게 변화하는지
# [torch-2] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-4
# python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-5
# python3 -m dropout.train --optimizer=momentum --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3
# python3 -m dropout.train --optimizer=momentum --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-4
# python3 -m dropout.train --optimizer=momentum --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-5
# python3 -m dropout.train --optimizer=sgd --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3
# python3 -m dropout.train --optimizer=sgd --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-4
# python3 -m dropout.train --optimizer=sgd --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-5

# RQ3. hidden layer 수에 따라서 어떻게 되는가?
# python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=1 --learning_rate=1e-3
# [torch-7] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=3 --learning_rate=1e-3
# [torch-7] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=4 --learning_rate=1e-3
# [torch-4] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=1 --learning_rate=1e-4
# [torch-5] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=3 --learning_rate=1e-4
# [torch-6] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=4 --learning_rate=1e-4

# RQ4. batch size에 어떻게 영향을 받는가?
# [torch-9] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=16
# [torch-8] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=32
# [torch-8] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=64
# python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=256
# python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=0

# RQ5. hidden unit 갯수에 어떻게 영향을 받는가?
# [torch-10] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=128 --n_hidden_unit=25
# [torch-10] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=128 --n_hidden_unit=100
# [torch-11] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=128 --n_hidden_unit=250
# [torch-11] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=128 --n_hidden_unit=500
# [torch-3] python3 -m dropout.train --optimizer=adam --activation_fn=relu --n_hidden_layer=2 --learning_rate=1e-3 --n_batch=128 --n_hidden_unit=1000

if __name__ == '__main__':
    main()
