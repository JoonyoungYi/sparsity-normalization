import os
import re
import sys


def main():
    """
        main module

        ex. python3 -m mnist.train ---n_hidden_unit_start=1 ---n_hidden_unit_end=100
    """
    args = []
    n_hidden_unit_start = None
    n_hidden_unit_end = None
    for arg in sys.argv[1:]:
        if arg.startswith('---'):
            m = re.search('---(?P<key>[\w\-\d\W]+)\=(?P<value>.*)$', arg)
            key, value = m.group('key'), m.group('value')
            if key == 'n_hidden_unit_start':
                n_hidden_unit_start = int(value.strip())
            elif key == 'n_hidden_unit_end':
                n_hidden_unit_end = int(value.strip())
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
    assert n_hidden_unit_start < n_hidden_unit_end

    commands = []
    for n_hidden_unit in range(n_hidden_unit_start, n_hidden_unit_end + 1):
        command = "python3 -m mnist.trainer --n_hidden_unit={n_hidden_unit} ".format(
            n_hidden_unit=n_hidden_unit) + ' '.join(args)
        commands.append(command)

    for command in commands:
        print('\n$', command)
        os.system(command)


if __name__ == '__main__':
    main()
