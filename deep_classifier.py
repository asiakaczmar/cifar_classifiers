import warnings
import itertools
from keras.applications import vgg16
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.models import Sequential
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from shallow_classifier import train_svm_classifier, evaluate_shallow_classifier


def preprocess_images(images):
    """
    Preprocess images for feature extraction. Subtract mean value
    (that was measured during training) and divide by 255.
    Then resize to 48 by 48 to make it a bit easier for the network
    which was trained on big images (256x256) and trained to detect
    features for those.
    :param images: np array
    :return: preprocessed images
    """
    #Values of means taken from tensoflow-> slim-> vgg_preprocessing.py
    images = images.astype(np.float32)
    means = [123.68, 116.779, 103.939]
    images -= means
    images /= 255
    with warnings.catch_warnings(record=True):
        #skimage throws a bunch of very unusefull warning
        #about the api changes
        images = np.array([resize(i, (48, 48, 3)) for i in images])
    return images


def get_feature_extraction_function(model, layer_number):
    """
    Create keras function that will input an image to a model, and
    get output from a given layer number
    """
    layers = [layer for layer in model.layers]
    layers = layers[:layer_number]
    layers.append(MaxPooling2D(pool_size=(4, 4)))
    model = Sequential(layers)
    return K.function([model.input], [model.output])


def get_features(model, images, params_dict):
    """
    Create and run feature_extraction function using params_dict
    """
    fe_function = get_feature_extraction_function(model, **params_dict)
    return fe_function([images])[0]


def train_and_evaluate_deep(model, train_images, test_images, train_labels,
                            test_labels, param_dict):
    """
    Run training and evaluation for given params

    """
    print('Trying params combination: {}'.format(str(param_dict)))
    print('getting features for training')
    fe_function = get_feature_extraction_function(model, **param_dict)
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_features = fe_function([train_images])[0]
    test_features = fe_function([test_images])[0]
    train_features = np.reshape(train_features, (train_features.shape[0], -1))
    test_features = np.reshape(test_features, (test_features.shape[0], -1))
    print('training svm classifier')
    classifier = train_svm_classifier(train_features, train_labels)
    print('evaluating')
    train_score = evaluate_shallow_classifier(classifier, train_features, train_labels)
    test_score = evaluate_shallow_classifier(classifier, test_features, test_labels)
    return train_score, test_score


def hyperparams_search_deep(model, images, labels, params_values):
    """
    Perform hyperparam search using gridsearch. Not very efficient.
    """
    param_names, vals_to_test = zip(*params_values.items())
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2)
    best_val_score = 0
    best_params = None
    for combination in itertools.product(*vals_to_test):
        param_dict = dict(zip(param_names, combination))
        train_score, val_score = train_and_evaluate_deep(model,train_images, val_images, train_labels,
                                                         val_labels, param_dict)
        if val_score > best_val_score:
            best_val_score = val_score
            best_params = combination
        print('The result was: {} for train and {} for val'.format(str(train_score), str(val_score)))
    return best_val_score, dict(zip(param_names, best_params))


def test_best_deep(model, train_images, train_labels, test_images,
                   test_labels, best_params):
    """
    Test using best parameters found on testing set
    """
    train_score, test_score = train_and_evaluate_deep(model, train_images,
                                                      test_images, train_labels,
                                                      test_labels, best_params)
    print('final evaluation with best params. Train score: ' +
          str(train_score) + ', test score: ' + str(test_score) + '.')
    return train_score, test_score


def train_pca(features):
    pca = PCA(n_components=2)
    pca.fit(features)
    return pca


def visualise_pc(pca, features, labels):
    """
    Visualise two most significant principal components in 2d.
    """
    data_2d = pca.transform(features)
    plt.figure(num=None, figsize=(20, 15))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10')
    plt.show()


def run_search_and_evaluation_deep(model, train_images, train_labels, test_images, test_labels, params_to_test):
    """
    run hyperparameter search and evaluate on test set.
    """
    best_val_score, best_params = hyperparams_search_deep(model, train_images, train_labels, params_to_test)
    train_score, test_score = test_best_deep(model, train_images, train_labels, test_images,
                                             test_labels, best_params)
    return test_score, best_params

def run_pca_visualisation(model, images, best_params, labels):
    """
    run PCA dimensionality reduction algorithm on features to reduce it to
    two most significant principal components. Visualise as a scatterplot.
    """
    print("preprocessing images")
    images = preprocess_images(images)
    print ("getting features")
    features = get_features(model, images, best_params)
    features = np.reshape(features, (features.shape[0], -1))
    print("getting pc")
    pca = train_pca(features)
    print("building visualisation")
    return visualise_pc(pca, features, labels)

