import os
import glob

from PIL import Image
import numpy as np

from ..utils import get_file_path


def _get_image_dict():
    image_dict = {}

    class_index = 0
    all_classes = glob.glob('.datasets/omniglot/images_background/*/*')
    for class_path in all_classes:
        all_images = glob.glob(class_path + '/*')
        assert len(all_images) == 20
        image_dict[class_index] = all_images
        class_index += 1

    all_classes = glob.glob('.datasets/omniglot/images_evaluation/*/*')
    for class_path in all_classes:
        all_images = glob.glob(class_path + '/*')
        assert len(all_images) == 20
        image_dict[class_index] = all_images
        class_index += 1

    return image_dict


def load_data(pixel=28):
    assert 28 <= pixel <= 105

    filepath = '.datasets/omniglot/x.npy'
    if os.path.exists(filepath):
        x = np.load(filepath)
    else:
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

        image_dict = _get_image_dict()
        x = np.zeros([len(image_dict.keys()), 20 * 4, pixel, pixel])
        for class_index, image_paths in tqdm(image_dict.items()):
            for j, image_path in enumerate(image_paths):
                image = Image.open(image_path)
                image = image.resize((pixel, pixel), resample=Image.LANCZOS)

                for angle in range(4):
                    image_array = np.array(image).astype(np.float32)
                    assert np.max(image_array) == 1
                    x[class_index][j + 20 * angle] = 1.0 - image_array
                    image = image.rotate(90)
        np.save(filepath, x)

    return x


# if __name__ == '__main__':
    # load_data()
