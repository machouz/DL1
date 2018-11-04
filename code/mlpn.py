import numpy as np
from loglinear import softmax
from utils import *

STUDENT={'name': 'cattana_',
         'ID': '336319314_F957022'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    out = x
    layer_number = len(params)
    for i in xrange(0, layer_number, 2):
        hidden = out.dot(params[i]) + params[i + 1]
        out = np.tanh(hidden)

    probs = softmax(out)

    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def hidden_layers(x, params):
    aggregation = []
    activation = [x]
    out = x

    layer_number = len(params)
    for i in xrange(0, layer_number, 2):
        aggr = out.dot(params[i]) + params[i + 1]
        out = np.tanh(aggr)
        aggregation.append(aggr)
        activation.append(out)

    y_pred = softmax(out)

    return y_pred, aggregation, activation

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE

    y_pred, aggregation, activation = hidden_layers(x, params)
    loss = -np.log(y_pred[y])
    grads = []



    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    layers_number = len(dims)
    params = []

    for i in xrange(layers_number - 1):
        eps = np.sqrt(6.0 / (dims[i] + dims[i+1]))
        W = np.random.uniform(-eps, eps, (dims[i], dims[i+1]))
        b = np.random.uniform(-eps, eps, dims[i+1])
        params.append(W)
        params.append(b)

    return params


if __name__ == '__main__':
    params = create_classifier([600, 200, 100, 6])
    x = np.random.uniform(0, 1, 600)
    y_pred, aggregation, activation = hidden_layers(x, params)