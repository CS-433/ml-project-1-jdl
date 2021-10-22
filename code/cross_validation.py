import numpy as np
import matplotlib.pyplot as plt

from helpers import compute_loss

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


def cross_validation(y, tx, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k])
    tx_te = tx[k_indices[k]]
    y_te = y[k_indices[k]]
    # choose desired method
    weights, _ = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_tr = compute_loss(y_tr, tx_tr, weights)
    loss_te = compute_loss(y_te, tx_te, weights)
    return loss_tr, loss_te


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