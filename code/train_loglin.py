import loglinear as ll
import random
import numpy as np
from utils import *

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

VOCAB_SIZE = len(vocab)
CATEGORIES = len(L2I)
num_iterations = 10
learning_rate = 0.01

def feats_to_vec(features):
    vec = np.zeros(VOCAB_SIZE)
    features = map(F2I.get, features)
    for feature in features:
        if feature != None:
            vec[feature] += 1

    # Should return a numpy vector of features.
    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y_pred = ll.predict(x, params)
        if y_pred == L2I.get(label):
            good += 1
        else:
            bad += 1
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    W, b = params
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I.get(label)                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            gW, gB = grads
            W -= learning_rate * gW
            b -= learning_rate * gB

            # YOUR CODE HERE
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
    out_dim = CATEGORIES
   

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(TRAIN, DEV, num_iterations, learning_rate, params)

