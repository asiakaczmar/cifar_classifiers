import os
import pickle
import requests
import tarfile

from settings import DATA_URL, LOCAL_DIR, DATA_DIRNAME, LOCAL_FILENAME


def maybe_download_cifar():
    local_filepath = os.path.join(LOCAL_DIR, DATA_DIRNAME)
    if os.path.exists(local_filepath):
        print('Already downloaded.')
    """
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    response = requests.get(DATA_URL, stream=True)
    with open(local_filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    tfile = tarfile.open(name=os.path.join(LOCAL_DIR, LOCAL_FILENAME), mode="r:gz")
    out_dir = os.path.join(LOCAL_DIR, DATA_DIRNAME)
    os.mkdir(out_dir)
    tfile.extractall(out_dir)
    # dTODO or whatever this is called :)
    os.rmtree(local_filepath)
    """


def read_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f, encoding='latin1')#, encoding='bytes')
