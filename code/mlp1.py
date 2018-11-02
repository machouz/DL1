import numpy as np
from utils import *

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}

CATEGORIES = len(L2I)


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    x -= np.max(x)
    exps = np.exp(x)
    return exps / np.sum(exps)


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a log-linear classifier with given params on input x.
    """
    [W, b, U, b_tag] = params
    hidden_output = x.dot(W) + b
    probs = hidden_output.dot(U) + b_tag
    return probs, hidden_output


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    probs, _ = classifier_output(x, params)
    return np.argmax(probs)


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    y_pred, hidden_output = classifier_output(x, params)  # y_pred:the prediction
    loss = -np.log(y_pred[y])  # log of the probability predicted to the correct class
    [W, b, U, b_tag] = params
    y_one_hot = one_hot_vector(y, len(b_tag))  # vector of zeros with one at the correct index
    gU = np.outer(hidden_output, y_pred - y_one_hot)  # [12:6]
    gb_tag = y_pred - y_one_hot  # [6]
    dt = 1.0 - hidden_output ** 2
    gW = np.outer(dt, x)  # [601,12]
    gb = dt  # [12]
    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """

    W = np.zeros((in_dim, hid_dim))
    b = np.zeros(hid_dim)
    U = np.zeros((hid_dim, out_dim))
    b_tag = np.zeros(out_dim)
    return [W, b, U, b_tag]

