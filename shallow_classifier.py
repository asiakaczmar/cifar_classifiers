import itertools

import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def get_features(image, orientations=9, cell_size=(4,4), cells_per_block=(4, 4)):
    """
    Uses the HOG algorithm to get features from an image to run a
    shallow classifier on top of those features
    :param image: numpy array
    :param orientations:
    :param cell_size:
    :param cells_per_block:
    :return: extracted features
    """
    return hog(image, orientations=orientations, pixels_per_cell=cell_size,
               cells_per_block=cells_per_block, visualize=True, multichannel=True,
               block_norm='L2')[0]


def get_processed_data(images, param_dict):
    """
    Just a helper function to run HOG feature extractor for all images
    in a list
    :param images: list of np.arrays
    :param param_dict: dictionary of params for HOG algorithm
    :return: np.array of extracted features
    """
    return np.stack([get_features(i, **param_dict) for i in images])


def train_svm_classifier(dataset, labels):
    """
    Fit a SVM classifier on a given dataset
    :param dataset: classifier input data
    :param labels: dataset labels. List of ints
    :return:
    """
    svm_classifier = SVC()
    return svm_classifier.fit(dataset, labels)


def evaluate_shallow_classifier(classifier, dataset, labels):
    """
    Evaluate a classifier
    :param classifier: trained classifier
    :param dataset: classifier evaluation data
    :param labels: evaluation data labels. List of ints
    :return: fraction of right guesses
    """
    predicted_labels = classifier.predict(dataset)
    return sum(predicted_labels == labels) / len(labels)


def train_and_evaluate_shallow(train_images, train_labels, test_images,
                               test_labels, param_dict):
    train_features = get_processed_data(train_images, param_dict)
    test_features = get_processed_data(test_images, param_dict)
    classifier = train_svm_classifier(train_features, train_labels)
    train_score = evaluate_shallow_classifier(classifier, train_features, train_labels)
    test_score = evaluate_shallow_classifier(classifier, test_features, test_labels)
    return train_score, test_score


def hyperparams_search_shallow(images, labels, params_values):
    """
    Run through given lists of possible values of parameters and perform
    grid search. Not very efficient.
    :param images: a list of np arrays
    :param labels: train dataset labels
    :param params_values: a dictionary, that has param name as a key
    and a list of possible values as value
    :return: a tuple: (best validation score, best_params)
    """
    param_names, vals_to_test = zip(*params_values.items())
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels,
                                                                          test_size=0.2)
    best_val_score = 0
    best_params = None
    for combination in itertools.product(*vals_to_test):
        param_dict = dict(zip(param_names, combination))
        train_score, val_score = train_and_evaluate_shallow(train_images, train_labels,
                                                            val_images, val_labels,
                                                            param_dict)
        if val_score > best_val_score:
            best_params = combination
        print(str(combination) + '->' + 'train score: ' + str(train_score) +
                                         'val_score: ' + str(val_score))
    print('best results (val_score: ' + str(best_val_score) + ' found for ' + str(best_params))
    return best_val_score, dict(zip(param_names, best_params))


def test_best_shallow(train_images, train_labels, test_images, test_labels, best_params):
    """
    Test chosen param on test set
    """

    train_score, test_score =  train_and_evaluate_shallow(train_images, train_labels,
                                       test_images, test_labels,
                                       best_params)
    print('final evaluation with best params. Train score: ' +
          str(train_score) + ', test score: ' + str(test_score) + '.')
    return train_score, test_score

def run_search_and_evaluation_shallow(train_images, train_labels, test_images,
                                      test_labels, params_to_test):
    """
    Perform hyperparam search and then evaluate a classifier trained on best
    params on a test set
    """
    best_val_score, best_params = hyperparams_search_shallow(train_images,
                                                             train_labels,
                                                             params_to_test)
    train_score, test_score = test_best_shallow(train_images, train_labels,
                                                test_images,test_labels,
                                                best_params)
    return test_score, best_params




