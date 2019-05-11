import os
import hashlib

from tqdm import tqdm


def _get_hash(path, blocksize=65536):
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def _download(file_path, url):
    import requests
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print("url이 유효하지 않습니다.")
        return False

    with open(file_path, 'wb') as f:
        for i, chunk in tqdm(enumerate(r.iter_content(1024))):
            f.write(chunk)
        f.close()
        return True


def get_file_path(filename, bucketname, url='', hash=None):
    assert filename
    assert url

    bucket_path = '.datasets/'
    if not os.path.exists(bucket_path):
        os.mkdir(bucket_path)

    dir_path = bucket_path + bucketname + '/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    file_path = dir_path + filename
    if not os.path.exists(file_path):
        success = _download(file_path, url)
        if not success:
            return None

    print('>> dataset hash ->', _get_hash(file_path))
    if hash:
        assert hash == _get_hash(file_path)

    return file_path
