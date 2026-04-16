import os
import glob

from PIL import Image
import numpy as np

from ..utils import get_file_path


def resize_data(pixel=28):
    assert 28 <= pixel <= 105

    train_dir_path = get_file_path(
        'images_background.zip',
        'omniglot',
        url=
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        hash='68d2efa1b9178cc56df9314c21c6e718',
        extract=True)
    # print(train_dir_path)

    test_dir_path = get_file_path(
        'images_evaluation.zip',
        'omniglot',
        url=
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip',
        hash='6b91aef0f799c5bb55b94e3f2daec811',
        extract=True)
    # print(test_dir_path)

    try:
        from tqdm import tqdm
    except:
        tqdm = lambda x: x

    image_paths = glob.glob(
        '.datasets/omniglot/images_background/*/*/*.png') + glob.glob(
            '.datasets/omniglot/images_evaluation/*/*/*.png')

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        image = image.resize((pixel, pixel), resample=Image.LANCZOS)
        image.save(
            '.datasets/omniglot/omniglot_resize/{}.png'.format(i),
            format='png')
    # return x


if __name__ == '__main__':
    resize_data()
