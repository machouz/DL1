import mlpn
import random
import numpy as np
from utils import *

STUDENT = {'name': 'Arie Cattan_Moche Uzan',
           'ID': '336319314_F957022'}

VOCAB_SIZE = len(vocab)
CATEGORIES = len(L2I)
num_iterations = 100
learning_rate = 0.001


def feats_to_vec(features):
    vec = np.zeros(VOCAB_SIZE)
    for feature in features:
        if feature in F2I:  # check that the feature is in the dictionary
            vec[F2I[feature]] += 1  # increment the index of the feature by 1
    # Should return a numpy vector of features.
    return vec


def accuracy_on_dataset(dataset, params):
    good = total = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)  # convert features to a vector.
        y = L2I.get(label)  # convert the label to number if needed.
        if mlpn.predict(x, params) == y:  # compare the prediction and the correct label
            good += 1
        total += 1
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
    return good / total


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = L2I.get(label)  # convert the label to number if needed.
            loss, gradients = mlpn.loss_and_gradients(x, y, params)
            cum_loss += loss
            for param, dparam in zip(params, gradients):
                param -= learning_rate * dparam
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...
    in_dim = VOCAB_SIZE
    hidden_dim = CATEGORIES * 2
    out_dim = CATEGORIES

    params = mlpn.create_classifier([600, 200, 100, 6])
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)
