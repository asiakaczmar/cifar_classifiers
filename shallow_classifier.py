import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC

from settings import HOG_ORIENTATIONS, HOG_CELL_SIZE


def get_features(image, orientations, cell_size, cells_per_block):
    return hog(image, orientations=orientations, pixels_per_cell=cells_per_block,
               cells_per_block=cells_per_block, visualize=True, multichannel=True,
               block_norm='L2')[0]


def get_processed_data(images):
    np.stack([get_features(i) for i in images])


def train_shallow_classifier(dataset, labels):
    svm_classifier = SVC()
    return svm_classifier.fit(dataset, labels)

def evaluate_shallow_classifier(classifier, dataset, labels):
    predicted_labels = classifier.predict(dataset)
    return sum(predicted_labels == labels) / len(labels)

def train_and_evaluate_shallow(train_images, train_labels, test_images, test_labels):
    train_features = get_processed_data(train_images)
    test_features = get_processed_data(test_images)
    classifier = train_shallow_classifier(train_features, train_labels)
    train_acc = evaluate_shallow_classifier(classifier, train_features, train_labels)
    test_acc = evaluate_shallow_classifier(classifier, test_features, test_labels)
    return train_acc, test_acc