import os
import pickle
import requests
import tarfile

from settings import DATA_URL, LOCAL_DIR, DATA_DIRNAME, LOCAL_ZIPPED

def inflate_file(local_tarfile, local_filepath):
    tfile = tarfile.open(name=local_tarfile, mode="r:gz")
    os.mkdir(local_filepath)
    tfile.extractall(local_filepath)
    os.remove(local_tarfile)

def maybe_download_cifar():
    """
    Download cifar data from a url specified in settings if it's not
    already present in a given location
    """
    local_filepath = os.path.join(LOCAL_DIR, DATA_DIRNAME)
    if os.path.exists(local_filepath):
        print('Already downloaded.')
        return
    if not os.path.exists(LOCAL_DIR):
        os.mkdir(LOCAL_DIR)
    print('Downloaded data not found. Downloading now...')
    response = requests.get(DATA_URL, stream=True)
    local_tarfile = os.path.join(LOCAL_DIR, LOCAL_ZIPPED)
    with open(local_tarfile, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    inflate_file(local_tarfile, local_filepath)
    print('Done')


def read_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f, encoding='latin1')#, encoding='bytes')
