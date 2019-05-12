import os
import re
import sys


def main():
    """
        main module

        ex.
        - default: python3 -m mnist.train --dataset=cifar100 --gpu_memory_fraction=0.14 --normalize_input=False ---n_hidden_unit_start=301 ---n_hidden_unit_end=325
        - normal-0: python3 -m mnist.train --dataset=cifar100 --gpu_memory_fraction=0.14 --normalize_input=True ---n_hidden_unit_start=301 ---n_hidden_unit_end=325
        - normal-0,1: python3 -m mnist.train --dataset=cifar100 --gpu_memory_fraction=0.14 --normalize_input=True --normalize_layer_indexes=1 ---n_hidden_unit_start=301 ---n_hidden_unit_end=325
        - normal-0,1,2: python3 -m mnist.train --dataset=cifar100 --gpu_memory_fraction=0.14 --normalize_input=True --normalize_layer_indexes=1,2 ---n_hidden_unit_start=301 ---n_hidden_unit_end=325
        - normal-1: python3 -m mnist.train --dataset=cifar100 --gpu_memory_fraction=0.14 --normalize_input=False --normalize_layer_indexes=1 ---n_hidden_unit_start=301 ---n_hidden_unit_end=325
        - normal-2: python3 -m mnist.train --dataset=cifar100 --gpu_memory_fraction=0.1 --normalize_input=False --normalize_layer_indexes=2 ---n_hidden_unit_start=201 ---n_hidden_unit_end=225
        - python3 -m mnist.train --dataset=cifar100 --gpu_memory_fraction=0.45 --normalize_input=False ---n_hidden_unit_start=310 ---n_hidden_unit_end=1000 ---n_hidden_unit_step=10
    """
    args = []
    n_hidden_unit_start = None
    n_hidden_unit_end = None
    n_hidden_unit_step = 1
    reverse = False
    for arg in sys.argv[1:]:
        if arg.startswith('---'):
            m = re.search('---(?P<key>[\w\-\d\W]+)\=(?P<value>.*)$', arg)
            key, value = m.group('key'), m.group('value')
            if key == 'n_hidden_unit_start':
                n_hidden_unit_start = int(value.strip())
            elif key == 'n_hidden_unit_end':
                n_hidden_unit_end = int(value.strip())
            elif key == 'n_hidden_unit_step':
                n_hidden_unit_step = int(value.strip())
            elif key == 'reverse':
                reverse = True
            else:
                raise NotImplementedError()
        elif arg.startswith('--'):
            for t in ['n_hidden_unit']:
                if arg.startswith('--' + t):
                    raise Exception('Do not pass argument {}'.format(t))
            args.append(arg)
        else:
            raise NotImplementedError()

    assert n_hidden_unit_start
    assert n_hidden_unit_end
    assert n_hidden_unit_step >= 1
    assert n_hidden_unit_start <= n_hidden_unit_end

    commands = []
    if reverse:
        g = range(n_hidden_unit_end, n_hidden_unit_start - 1,
                  -n_hidden_unit_step)
    else:
        g = range(n_hidden_unit_start, n_hidden_unit_end, n_hidden_unit_step)
    for n_hidden_unit in g:
        command = "python3 -m mnist.trainer --n_hidden_unit={n_hidden_unit} ".format(
            n_hidden_unit=n_hidden_unit) + ' '.join(args)
        commands.append(command)

    for command in commands:
        print('\n$', command)
        os.system(command)


if __name__ == '__main__':
    main()
