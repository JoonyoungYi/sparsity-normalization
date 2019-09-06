from common.parallel import run_parallel

if __name__ == '__main__':
    commands = [
        'python3 -m lenet.trainer --dropout_rate=0.1',
        'python3 -m lenet.trainer --dropout_rate=0.2',
        'python3 -m lenet.trainer --dropout_rate=0.3',
        'python3 -m lenet.trainer --dropout_rate=0.4',
        'python3 -m lenet.trainer --dropout_rate=0.5',
        'python3 -m lenet.trainer --dropout_rate=0.6',
        'python3 -m lenet.trainer --dropout_rate=0.7',
        'python3 -m lenet.trainer --dropout_rate=0.8',
        'python3 -m lenet.trainer --dropout_rate=0.9',
    ]
    run_parallel(commands, n_worker=9)
