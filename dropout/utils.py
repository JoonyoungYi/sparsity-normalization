import os


def dict2tsv(res, file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'a') as f:
            f.write('\t'.join(list(res.keys())))
            f.write('\n')

    with open(file_name, 'a') as f:
        f.write('\t'.join([str(r) for r in list(res.values())]))
        f.write('\n')


if __name__ == '__main__':
    import collections
    import random

    for epoch in range(100):
        loss = random.random()
        acc = random.random()

        res = collections.OrderedDict()
        res['epoch'] = epoch
        res['loss'] = loss
        res['acc'] = acc

        dict2tsv(res, 'result.txt')
