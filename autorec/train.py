import os
import re
import sys


def main():
    """
        main module

        ex. python3 -m autorec.train --use_user_vector=False --learning_rate=1e-3 --optimizer=adam --dataset=movielens-100k --sparse_tensor=True --n_batch=0 ---exp=default ---lambdas=100,200,300,400,500 ---seeds=2,3,4,5
    """
    args = []
    exp = None
    seeds = None
    lambdas = None
    for arg in sys.argv[1:]:
        if arg.startswith('---'):
            m = re.search('---(?P<key>[\w\-\d\W]+)\=(?P<value>.*)$', arg)
            key, value = m.group('key'), m.group('value')
            if key == 'exp':
                exp = value
                assert value in ['default', 'default-2', 'normal-2']
            elif key == 'lambdas':
                lambdas = [float(t) for t in value.split(',')]
            elif key == 'seeds':
                seeds = [int(t) for t in value.split(',')]
        elif arg.startswith('--'):
            for t in ['normalize_input', 'seed', 'lambda_w1', 'lambda_w2']:
                if arg.startswith('--' + t):
                    raise Exception('Do not pass argument {}'.format(t))
            args.append(arg)
        else:
            raise NotImplementedError()

    assert exp
    assert lambdas
    assert seeds

    commands = []
    if exp == 'default':
        for l in lambdas:
            for seed in seeds:
                command = "python3 -m autorec.trainer --normalize_input=False --seed={seed} --lambda_w1={l} --lambda_w2={l} ".format(
                    seed=seed, l=l) + ' '.join(args)
                commands.append(command)
    elif exp == 'default-2':
        for l in lambdas:
            for seed in seeds:
                command = "python3 -m autorec.trainer --normalize_input=False --seed={seed} --lambda_w1=0.0 --lambda_w2={l} ".format(
                    seed=seed, l=l) + ' '.join(args)
                commands.append(command)
    elif exp == 'normal-2':
        for l in lambdas:
            for seed in seeds:
                command = "python3 -m autorec.trainer --normalize_input=True --seed={seed} --lambda_w1=0.0 --lambda_w2={l} ".format(
                    seed=seed, l=l) + ' '.join(args)
                commands.append(command)
    else:
        raise NotImplementedError()
    # os.system(command)

    for command in commands:
        print('\n$', command)
        os.system(command)


if __name__ == '__main__':
    main()
