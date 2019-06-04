import os
import re
import sys


def main():
    """
        main module

        ex.
        python3 -m uci.train ---n_replications=1,2,3,4,5,6,7,8,9,10
    """
    args = []
    exp = None
    seeds = None
    n_replications = None
    for arg in sys.argv[1:]:
        if arg.startswith('---'):
            m = re.search('---(?P<key>[\w\-\d\W]+)\=(?P<value>.*)$', arg)
            key, value = m.group('key'), m.group('value')
            if key == 'n_replications':
                n_replications = [int(t) for t in value.split(',')]
            elif key == 'seeds':
                seeds = [int(t) for t in value.split(',')]
            else:
                raise NotImplementedError()
        elif arg.startswith('--'):
            for t in ['seed', 'n_replication']:
                if arg.startswith('--' + t):
                    raise Exception('Do not pass argument {}'.format(t))
            args.append(arg)
        else:
            raise NotImplementedError()

    assert lambdas
    assert seeds

    commands = []
    for seed in seeds:
        for n_replication in n_replications:
            command = "python3 -m uci.trainer --seed={seed} --n_replication={n_replication} ".format(
                seed=seed, n_replication=n_replication) + ' '.join(args)
            commands.append(command)

    for command in commands:
        print('\n$', command)
        os.system(command)


if __name__ == '__main__':
    main()
