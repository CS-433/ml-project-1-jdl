import numpy as np


# Compute gradient and error vector
def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    n = len(y)

    gradient = -1 / n * tx.T.dot(e)
    return gradient, e


# Functions for the calculation of the loss
def calculate_mse(e):
    return 1 / 2 * np.mean(e ** 2)


# def calculate_mae(e):
#     return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)


# Functions used fo the logistic regression
def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1./(1 + np.exp(-t))


def calculate_sigmoid_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sig = sigmoid(tx.dot(w))
    term1 = (-1)*y.T.dot(np.log(sig))
    term2 = (-1)*(1 - y).T.dot(np.log(1 - sig))
    return term1 + term2


def calculate_sigmoid_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig = sigmoid(tx.dot(w))
    return tx.T.dot(sig - y)


def calculate_sigmoid_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    sig = sigmoid(tx.dot(w))
    S = np.diag(sig.ravel()*(1-sig.ravel()))
    return tx.T.dot(S).dot(tx)


def help_logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""
    loss = calculate_sigmoid_loss(y, tx, w)
    gradient = calculate_sigmoid_gradient(y, tx, w)
    hessian = calculate_sigmoid_hessian(y, tx, w)
    return loss, gradient, hessian


def help_penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_sigmoid_loss(y, tx, w) + lambda_ * np.linalg.norm(w, 2)**2
    gradient = calculate_sigmoid_gradient(y, tx, w) + 2*lambda_*w
    hessian = calculate_sigmoid_hessian(y, tx, w) + 2*lambda_
    return loss, gradient, hessian


# Function for loading the data
def load_data(dataset_type='train'):
    dataset_type = dataset_type.lower().strip()
    if dataset_type == 'train':
        path_dataset = "/Users/loicomeliau/Documents/EPFL/courses/machine_learning/project1/datasets/train.csv"
    elif dataset_type == 'test':
        # Path to adapt because no in the repo
        path_dataset = "../data/test.csv"
    feature_cols = np.arange(32)[2:]
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=feature_cols)
    features = data
    labels = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1], dtype=bytes,
        converters={1: lambda x: 0 if b"s" in x else 1})
    return features, labels


# Function to create batch iter for SGD
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def correctness(train_tx, train_y, test_tx, test_y, weights):
    # Make predictions
    train_pred = train_tx.dot(weights)
    test_pred = test_tx.dot(weights)

    # Transform the prediction into 0 ( = 's') and 1 (= 'b')
    train_pred = np.where(train_pred > 0.5, 1, 0)
    test_pred = np.where(test_pred > 0.5, 1, 0)

    # Compute the ratio of correct labeled predictions
    train_score = np.sum(np.where(train_pred == train_y, 1, 0)) / len(train_pred)
    test_score = np.sum(np.where(test_pred == test_y, 1, 0)) / len(test_pred)

    print("There are {train_s}% correct prediction in the training set".format(train_s=train_score * 100))
    print("There are {test_s}% correct prediction in the test set".format(test_s=test_score * 100))

    return train_score, test_score
