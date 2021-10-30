import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import compute_loss, calculate_sigmoid_loss, accuracy

from implementations import least_squares_GD, least_squares_SGD, \
    ridge_regression, logistic_regression, reg_logistic_regression


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cv_GD(y, tx, k_indices, k, max_iters, gamma):
    """return the loss of ridge regression."""
    initial_w = np.ones(len(tx[0])) * 0.1
    # get k'th subgroup in test, others in train
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k])
    tx_te = tx[k_indices[k]]
    y_te = y[k_indices[k]]
    # choose desired method
    weights, _ = least_squares_GD(y, tx, initial_w, max_iters, gamma)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, weights))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, weights))
    # calculate accuracy
    acc = accuracy(tx_tr, y_tr, tx_te, y_te, weights, print_=False)
    return loss_tr, loss_te, acc


def cv_SGD(y, tx, k_indices, k, max_iters, gamma):
    """return the loss of ridge regression."""
    initial_w = np.ones(len(tx[0])) * 0.1
    # get k'th subgroup in test, others in train
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k])
    tx_te = tx[k_indices[k]]
    y_te = y[k_indices[k]]
    # choose desired method
    weights, _ = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, weights))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, weights))
    # calculate accuracy
    acc = accuracy(tx_tr, y_tr, tx_te, y_te, weights, print_=False)
    return loss_tr, loss_te, acc


def cv_ridge_regression(y, tx, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k])
    tx_te = tx[k_indices[k]]
    y_te = y[k_indices[k]]
    # choose desired method
    weights, _ = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, weights))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, weights))
    # calculate accuracy
    acc = accuracy(tx_tr, y_tr, tx_te, y_te, weights, print_=False)
    return loss_tr, loss_te, acc


def cv_logistic_regression(y, tx, k_indices, k, max_iters, gamma):
    """return the loss of ridge regression."""
    initial_w = np.ones(len(tx[0])) * 0.1
    # get k'th subgroup in test, others in train
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k])
    tx_te = tx[k_indices[k]]
    y_te = y[k_indices[k]]
    # choose desired method
    weights, _ = logistic_regression(y, tx, initial_w, max_iters, gamma)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, weights))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, weights))
    # calculate accuracy
    acc = accuracy(tx_tr, y_tr, tx_te, y_te, weights, print_=False)
    return loss_tr, loss_te, acc


def cv_reg_logistic_regression(y, tx, k_indices, k, lambda_, max_iters, gamma):
    """return the loss of ridge regression."""
    initial_w = np.ones(len(tx[0])) * 0.1
    # get k'th subgroup in test, others in train
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k])
    tx_te = tx[k_indices[k]]
    y_te = y[k_indices[k]]
    # choose desired method
    weights, _ = reg_logistic_regression(y, tx, initial_w, lambda_, max_iters, gamma)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * calculate_sigmoid_loss(y_tr, tx_tr, weights))
    loss_te = np.sqrt(2 * calculate_sigmoid_loss(y_te, tx_te, weights))
    # calculate accuracy
    acc = accuracy(tx_tr, y_tr, tx_te, y_te, weights, print_=False)
    return loss_tr, loss_te, acc


def cross_validation_lambda_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.xlim(1e-4, 1)
    plt.title("cross validation lambda")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_lambda")


def cross_validation_gamma_visualization(gams, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(gams, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(gams, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation gamma")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_gamma")