# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """
    Loads data

    Parameters
    ----------
    data_path : str | Path
        path of the data that needs to be loaded
    sub_sample : bool
        sub_sample flag

    Returns
    -------
    y : binary array
        class labels
    tX : float array
        features matrix
    ids : int array
        event ids
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # Convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = 0
    
    # Sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """
    Generates class predictions.

    Parameters
    ----------
    weights : float array
        weight vector used to build the prediction
    data : float array
        test data matrix used to build the prediction

    Returns
    -------
    y_pred : binary array
        prediction
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd.

    Parameters
    ----------
    ids : int array
        event ids associated with each prediction
    y_pred : binary array
        predicted class labels
    name : str
        string name of .csv output file to be created
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def compute_gradient(y, tx, w):
    """
    Compute gradient and error vector.

    Parameters
    ----------
    y : binary array
        class labels
    tx : float array
        feature matrix
    w : float array
        weight vector

    Returns
    -------
    gradient : float array
        gradient vector
    e : float array
        error vector
    """
    e = y - tx.dot(w)
    n = len(y)

    gradient = -tx.T.dot(e) / n
    return gradient, e


def calculate_mse(e):
    """
    Compute mean square error.

    Parameters
    ----------
    e : float array
        error vector

    Returns
    -------
    mse : float
        mean square error
    """
    mse = 1 / 2 * np.mean(e ** 2)
    return mse


def compute_loss(y, tx, w):
    """
    Compute loss (using mse).

    Parameters
    ----------
    y : binary array
        class labels
    tx : float array
        feature matrix
    w : float array
        weight vector

    Returns
    -------
    loss : float
        loss (mean square error)
    """
    e = y - tx.dot(w)
    loss = calculate_mse(e)
    return loss


# Functions used fo the logistic regression
def sigmoid(t):
    """
    Apply sigmoid function.

    Parameters
    ----------
    t : float array
        input of the sigmoid

    Returns
    -------
    sigmoid function
    """
    tt = t.copy()
    tt[tt < -100] = -100
    tt[tt > 19] = 19
    return 1./(1 + np.exp(-tt))


def calculate_sigmoid_loss(y, tx, w):
    """
    Compute the loss: negative log likelihood.

    Parameters
    ----------
    y : binary array
        class labels
    tx : float array
        feature matrix
    w : float array
        weight vector

    Returns
    -------
    loss : float
        loss (negative log likelihood)
    """
    sig = sigmoid(tx.dot(w))
    term1 = (-1) * y.T.dot(np.log(sig))
    term2 = (-1) * (1 - y).T.dot(np.log(1 - sig))
    loss = term1 + term2
    return loss


def calculate_sigmoid_gradient(y, tx, w):
    """
    Compute the gradient of (sigmoid) loss.

    Parameters
    ----------
    y : binary array
        class labels
    tx : float array
        feature matrix
    w : float array
        weight vector

    Returns
    -------
    grad : float array
        gradient vector
    """
    sig = sigmoid(tx.dot(w))
    grad = tx.T.dot(sig - y)
    return grad


def calculate_sigmoid_hessian(y, tx, w):
    """
    Compute the Hessian of the loss function.

    Parameters
    ----------
    y : binary array
        class labels
    tx : float array
        feature matrix
    w : float array
        weight vector

    Returns
    -------
    hess : float array
        hessian matrix
    """
    sig = sigmoid(tx.dot(w))
    S = np.diag(sig.ravel()*(1-sig.ravel()))
    hess = tx.T.dot(S).dot(tx)
    return hess


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


def accuracy(train_tx, train_y, val_tx, val_y, weights, print_=True):
    """
    Compute train and validation accuracy.

    Parameters
    ----------
    train_tx : float array
        train feature matrix
    train_y : float array
        train class labels
    val_tx : float array
        validation feature matrix
    val_y : float array
        validation class labels
    weights : float array
        weight vector
    print : bool
        print flag

    Returns
    -------
    train_score : float
        train accuracy
    val_score : float
        validation accuracy
    """
    # Make predictions
    train_pred = predict_labels(weights, train_tx)
    val_pred = predict_labels(weights, val_tx)

    # Compute the ratio of correct labeled predictions
    train_score = np.sum(np.where(train_pred == train_y, 1, 0)) / len(train_pred)
    val_score = np.sum(np.where(val_pred == val_y, 1, 0)) / len(val_pred)

    if print_:
        print("There are {train_s}% correct prediction in the training set".format(train_s=train_score * 100))
        print("There are {val_s}% correct prediction in the validation set".format(val_s=val_score * 100))

    return train_score, val_score


def build_poly_tx(tx, idx, degree):
    """
    Polynomial basis function which applies polynomial expansion
    to indices in idx.

    Parameters
    ----------
    tx : float array
        feature matrix to expand
    idx : int array
        indices of feature to expand
    degree : int
        degree of polynomial expansion

    Returns
    -------
    new_tx : float array
        feature matrix with polynomial expansion
    """
    if degree == 0:
        return np.delete(tx, idx, axis = 1)
    elif degree == 1:
        return tx
    else:
        new_tx = tx.copy()
        new_tx = np.delete(new_tx, idx, axis=1)
        for i in idx:
            poly = np.power(tx[:, i], 1)
            for deg in range(2, degree+1):
                poly = np.c_[poly, np.power(tx[:, i], deg)]
            new_tx = np.hstack((new_tx, poly))
        return new_tx


def build_final_poly_tx(tx, best_degrees):
    """
    Compute the polynomial expansion of each features to its best
    degree.

    Parameters
    ----------
    tx : float array
        feature matrix
    best_degree : int array
        best degree of each feature

    Returns
    -------
    new_tx : float array
        feature matrix with polynomial expansion
    """
    new_tx = np.ones((len(tx[:, 0]), 1))
    for i, degree in enumerate(best_degrees):
        poly = np.power(tx[:, i+1], 1)
        for deg in range(2, degree+1):
            poly = np.c_[poly, np.power(tx[:, i+1], deg)]
        new_tx = np.c_[new_tx, poly]

    return new_tx
