import numpy as np
from utils import *

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


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
    z = hidden_output.dot(U) + b_tag
    probs = softmax(z)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    probs = classifier_output(x, params)
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


    y_pred = classifier_output(x, params)  # y_pred:the prediction
    loss = -np.log(y_pred[y])  # log of the probability predicted to the correct class
    [W, b, U, b_tag] = params
    hidden_output = x.dot(W) + b
    y_one_hot = one_hot_vector(y, len(b_tag))  # vector of zeros with one at the correct index
    gU = np.outer(hidden_output, y_pred - y_one_hot)  # [12:6]
    gb_tag = y_pred - y_one_hot  # [6]
    dl_z1 = U.dot(y_pred - y_one_hot)
    dz1_h1 = 1.0 - np.tanh(hidden_output) ** 2
    dl_h1 = dl_z1 * dz1_h1
    gW = np.outer(x, dl_h1)  # [600,12]
    gb = dl_h1  # [12]
    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """

    eps = np.sqrt(6.0 / (in_dim + hid_dim))
    W = np.random.uniform(-eps, eps, (in_dim, hid_dim))
    b = np.random.uniform(-eps, eps, hid_dim)

    eps = np.sqrt(6.0 / (hid_dim + out_dim))
    U = np.random.uniform(-eps, eps, (hid_dim, out_dim))
    b_tag = np.random.uniform(-eps, eps, out_dim)

    params = [W, b, U, b_tag]
    return params
