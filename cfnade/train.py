import os
import re
import sys


def main():
    """
        main module

        ex. python3 -m cfnade.trainer --dataset=movielens-100k --gpu_memory_fraction=0.05 --learning_rate=0.001 --use_user_vector=True --parameter_sharing=False --use_ordinal_cost=False --normalize_input=True --averaging_possible_choices=True ---exp=normal-2 ---lambdas=0.00001,0.00004,0.00007,0.0001,0.0004,,0.0007,0.001,0.004,0.007,0.01,0.04,0.07,0.1
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
            else:
                raise NotImplementedError()
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
