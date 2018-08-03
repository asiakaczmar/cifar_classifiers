import os

home = os.path.expanduser("~")

# Data loading
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
LOCAL_DIR = os.path.join(home, 'PhD', 'tooploox')
LOCAL_ZIPPED = 'cifar_dataset.tar.gz'
DATA_DIRNAME = 'cifar_data'
LOCAL_FILENAME = 'cifar_dataset.tar.gz'

# Image related
N_IMAGES_TO_SHOW = 10
CLASSES_NO = 10
IMAGE_W = 32
IMAGE_H = 32
CHANNELS = 3
# number of images from each class that we will use
N_IMAGES_PER_CLASS = 500

#convenience vars
TRAIN_DATA_FILENAME = os.path.join(LOCAL_DIR, DATA_DIRNAME, 'cifar-10-batches-py', 'data_batch_1')
TEST_DATA_FILENAME = os.path.join(LOCAL_DIR, DATA_DIRNAME, 'cifar-10-batches-py', 'test_batch')

#shallow classifier settings
HOG_ORIENTATIONS = 8
HOG_CELL_SIZE = (4, 4)