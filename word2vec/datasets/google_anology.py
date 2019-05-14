# word2vec 논문에서 사용한 데이터 셋.
# Reference: https://aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art)
import numpy as np

from common.dataset_utils import get_file_path


def _get_token_ids(tokens, word_to_id):
    token_ids = [word_to_id.get(t) for t in tokens]
    if sum(1 if t else 0 for t in token_ids) == 4:
        return token_ids


def load_data(word_to_id):
    file_path = get_file_path(
        'questions-words.txt',
        'google-analogy',
        url='http://download.tensorflow.org/data/questions-words.txt',
        hash='8b7461cbf7ecc0aec9b32eb626821105')

    semantic_data = []
    syntactic_data = []

    is_semantic = True
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                if line.startswith(": gram"):
                    is_semantic = False
                continue

            tokens = line.split(' ')
            assert len(tokens) == 4

            token_ids = _get_token_ids(tokens, word_to_id)
            if not token_ids:
                continue

            if is_semantic:
                semantic_data.append(token_ids)
            else:
                print(tokens)
                print(token_ids)
                syntactic_data.append(token_ids)
        f.close()
    input('')
    return np.array(semantic_data), np.array(syntactic_data)
