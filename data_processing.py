import numpy as np
from numpy.random import choice


from settings import (CHANNELS, IMAGE_W, IMAGE_H, N_IMAGES_PER_CLASS,
                      N_IMAGES_TO_SHOW, CLASSES_NO)


def get_labels(data):
    labels = data['labels']
    return np.array(labels)


def get_images_and_labels(data, shuffle=True):
    labels = get_labels(data)
    images = np.concatenate([get_random_n_images(data['data'], labels, c, N_IMAGES_PER_CLASS)
                             for c in range(10)])
    labels = np.concatenate([[c] * N_IMAGES_PER_CLASS for c in range(10)])
    images = np.array([process_image(i) for i in images])
    if shuffle:
        #TODO: take a look a this - is shuffle in place?
        permutation = np.random.shuffle(list(range(len(images))))
        images = images[permutation]
        labels = labels[permutation]
    return images, labels


def process_image(i):
    i = i.reshape(CHANNELS, IMAGE_W, IMAGE_H)
    return np.transpose(i, [1, 2, 0])


def get_random_n_images(images, labels, cls, no_images):
    class_images = images[labels == cls]
    chosen_indices = choice(range(N_IMAGES_PER_CLASS), no_images, replace=False)
    return class_images[chosen_indices]


def get_images_to_show(images, labels):
    return np.concatenate([get_random_n_images(images, labels, x, N_IMAGES_TO_SHOW)
                           for x in range(CLASSES_NO)])


