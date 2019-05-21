import os
import re
import sys


def main():
    """
        main module

        ex.
        python3 -m autorec_2.train --use_user_vector=False --learning_rate=1e-3 --optimizer=adam --dataset=movielens-100k --sparse_tensor=True --n_batch=0 ---lambdas=100,200,300,400,500 ---seeds=2,3,4,5 --filename=item-default --normalize_input=False
        python3 -m autorec_2.train --use_user_vector=False --learning_rate=1e-3 --optimizer=adam --dataset=movielens-100k --sparse_tensor=True --n_batch=0 ---lambdas=400,500,600,700,800 ---seeds=2,3,4,5 --filename=item-normal --normalize_input=True
        python3 -m autorec_2.train --use_user_vector=True --learning_rate=1e-3 --optimizer=adam --dataset=movielens-100k --sparse_tensor=True --n_batch=0 ---lambdas=100,200,300,400,500 ---seeds=2,3,4,5 --filename=user-default --normalize_input=False
        python3 -m autorec_2.train --use_user_vector=True --learning_rate=1e-3 --optimizer=adam --dataset=movielens-100k --sparse_tensor=True --n_batch=0 ---lambdas=0,1,4,7,10,40,70,100,400,700,1000 ---seeds=1 --filename=user-normal-explore --normalize_input=True
        python3 -m autorec_2.train --use_user_vector=True --learning_rate=1e-3 --optimizer=adam --dataset=movielens-100k --sparse_tensor=True --n_batch=0 ---lambdas=400,500,600,700,800 ---seeds=2,3,4,5 --filename=item-normal --normalize_input=True

        python3 -m autorec_2.train --use_user_vector=False --learning_rate=1e-3 --optimizer=adam --dataset=movielens-1m --sparse_tensor=True --n_batch=0 ---lambdas=300,400,500,600,700 ---seeds=2,3,4,5 --filename=item-default --normalize_input=False
        python3 -m autorec_2.train --use_user_vector=False --learning_rate=1e-3 --optimizer=adam --dataset=movielens-1m --sparse_tensor=True --n_batch=0 ---lambdas=0,1,4,7,10,40,70,100,400,700,1000 ---seeds=1 --filename=item-normal-explore --normalize_input=True
        python3 -m autorec_2.train --use_user_vector=False --learning_rate=1e-3 --optimizer=adam --dataset=movielens-1m --sparse_tensor=True --n_batch=0 ---lambdas=200,300,500,600,800,900 ---seeds=1 --filename=item-normal-explore --normalize_input=True
        python3 -m autorec_2.train --use_user_vector=True --learning_rate=1e-3 --optimizer=adam --dataset=movielens-1m --sparse_tensor=True --n_batch=0 ---lambdas=500,600,700,800,900 ---seeds=2,3,4,5 --filename=user-default --normalize_input=False
        python3 -m autorec_2.train --use_user_vector=True --learning_rate=1e-3 --optimizer=adam --dataset=movielens-1m --sparse_tensor=True --n_batch=0 ---lambdas=0,1,4,7,10,40,70,100,400,700,1000 ---seeds=1 --filename=user-normal-explore --normalize_input=True
        python3 -m autorec_2.train --use_user_vector=True --learning_rate=1e-3 --optimizer=adam --dataset=movielens-1m --sparse_tensor=True --n_batch=0 ---lambdas=200,300,500,600,800,900 ---seeds=1 --filename=user-normal-explore --normalize_input=True

        python3 -m autorec_2.train --use_user_vector=False --learning_rate=1e-4 --optimizer=adam --dataset=movielens-10m --sparse_tensor=True --n_batch=1000 ---lambdas=30,40,50,60,70 ---seeds=2,3,4,5 --filename=item-default --normalize_input=False --epoch_early_stop=10000
        python3 -m autorec_2.train --use_user_vector=True --learning_rate=1e-4 --optimizer=adam --dataset=movielens-10m --sparse_tensor=True --n_batch=1000 ---lambdas=50,60,70,80,90 ---seeds=2,3,4,5 --filename=user-default --normalize_input=False --epoch_early_stop=10000
    """
    args = []
    exp = None
    seeds = None
    lambdas = None
    for arg in sys.argv[1:]:
        if arg.startswith('---'):
            m = re.search('---(?P<key>[\w\-\d\W]+)\=(?P<value>.*)$', arg)
            key, value = m.group('key'), m.group('value')
            if key == 'lambdas':
                lambdas = [float(t) for t in value.split(',')]
            elif key == 'seeds':
                seeds = [int(t) for t in value.split(',')]
            else:
                raise NotImplementedError()
        elif arg.startswith('--'):
            for t in ['seed', 'lambda_w']:
                if arg.startswith('--' + t):
                    raise Exception('Do not pass argument {}'.format(t))
            args.append(arg)
        else:
            raise NotImplementedError()

    assert lambdas
    assert seeds

    commands = []
    for seed in seeds:
        for l in lambdas:
            command = "python3 -m autorec_2.trainer --seed={seed} --lambda_w={l} ".format(
                seed=seed, l=l) + ' '.join(args)
            commands.append(command)

    for command in commands:
        print('\n$', command)
        os.system(command)


if __name__ == '__main__':
    main()
