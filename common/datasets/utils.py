import os
import hashlib
import tarfile
import zipfile


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

    try:
        from tqdm import tqdm
    except:
        tqdm = lambda x: x
    with open(file_path, 'wb') as f:
        for i, chunk in tqdm(enumerate(r.iter_content(1024))):
            f.write(chunk)
        f.close()
        return True


def _extract_all(file_path, dir_path):
    if file_path.endswith(".tar.gz") or file_path.endswith('.tgz'):
        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(dir_path)
        tar.close()
    elif file_path.endswith(".tar"):
        tar = tarfile.open(file_path, "r:")
        tar.extractall(dir_path)
        tar.close()
    elif file_path.endswith(".zip"):
        zip = zipfile.ZipFile(file_path)
        zip.extractall(dir_path)
        zip.close()
    else:
        raise NotImplementedError()


def get_file_path(filename, bucketname, url='', hash=None, extract=False):
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

    if extract:
        print(file_path)
        _extract_all(file_path, dir_path)
        return dir_path
    else:
        return file_path
