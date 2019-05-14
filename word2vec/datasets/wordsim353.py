# GloVe 논문에서 word similarity로 사용한 데이터셋.
# 이외에도 수많은 논문에서 사용하고 있음.
# Word Embedding의 성능을 보통은 word similarity로 측정하는 것으로 파악.
import numpy as np

from common.dataset_utils import get_file_path


def _get_token_ids(tokens, word_to_id):
    token_ids = [word_to_id.get(t) for t in tokens]
    if sum(1 if t else 0 for t in token_ids) == 4:
        return token_ids


def load_data(word_to_id):
    file_path = get_file_path(
        'combined.csv',
        'wordsim-353',
        url=
        'https://raw.githubusercontent.com/JoonyoungYi/datasets/master/wordsim353/combined.csv',
        hash='5a605187d8c46fd27be60f0033c217a0')

    test_labels_a = []
    test_labels_b = []
    test_scores = []
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            line = line.strip()
            tokens = line.split(',')
            assert len(tokens) == 3

            # labels_a[]
            label_a = word_to_id.get(tokens[0])
            label_b = word_to_id.get(tokens[1])

            if label_a is None or label_b is None:
                continue

            # print(tokens)
            test_labels_a.append(label_a)
            test_labels_b.append(label_b)
            test_scores.append(float(tokens[2]))
        f.close()

    # print(len(test_scores))
    # input('')
    return np.array(test_labels_a), np.array(test_labels_b), np.array(
        test_scores)
