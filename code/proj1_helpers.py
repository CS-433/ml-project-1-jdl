# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


# Compute gradient and error vector
def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    n = len(y)

    gradient = -tx.T.dot(e) / n
    return gradient, e


# Functions for the calculation of the loss
def calculate_mse(e):
    return 1 / 2 * np.mean(e ** 2)


def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)


# Functions used fo the logistic regression
def sigmoid(t):
    """apply the sigmoid function on t."""
    tt = t.copy()
    tt[tt < -100] = -100
    tt[tt > 19] = 19
    return 1./(1 + np.exp(-tt))


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


def accuracy(train_tx, train_y, val_tx, val_y, weights, print_ = True):
    # Make predictions
    train_pred = predict_labels(weights, train_tx)
    val_pred = predict_labels(weights, val_tx)

    # Compute the ratio of correct labled predictions
    train_score = np.sum(np.where(train_pred == train_y, 1, 0)) / len(train_pred)
    val_score = np.sum(np.where(val_pred == val_y, 1, 0)) / len(val_pred)

    if print_:
        print("There are {train_s}% correct prediction in the training set".format(train_s=train_score * 100))
        print("There are {val_s}% correct prediction in the validation set".format(val_s=val_score * 100))

    return train_score, val_score
