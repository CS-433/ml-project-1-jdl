import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import compute_loss, calculate_sigmoid_loss, accuracy, build_poly_tx

from implementations import least_squares, least_squares_GD, least_squares_SGD, \
    ridge_regression, logistic_regression, reg_logistic_regression


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.

    Parameters
    ----------
    y : array
        array to separate
    k_fold : int
        number of folds
    seed : int
        seed

    Returns
    --------
    array of folded indices
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cv_GD(y, tx, k_indices, k, max_iters, gamma):
    """
    Cross validation for the gradient descent algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    k_indices : int array
        array of folded indices
    k : int
        index of the subset of index to form the test set
    max_iters : int
        number of iterations
    gamma : float
        step size

    Returns
    -------
    loss_tr : float
        training rmse
    loss_te : float
        test rmse
    acc : float array
        accuracy (train and test)
    """
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
    """
    Cross validation for the stochastic gradient descent algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    k_indices : int array
        array of folded indices
    k : int
        index of the subset of index to form the test set
    max_iters : int
        number of iterations
    gamma : float
        step size

    Returns
    -------
    loss_tr : float
        training rmse
    loss_te : float
        test rmse
    acc : float array
        accuracy (train and test)
    """
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
    """
    Cross validation for the ridge regression algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    k_indices : int array
        array of folded indices
    k : int
        index of the subset of index to form the test set
    lambda_ : float
        regularisation parameter

    Returns
    -------
    loss_tr : float
        training rmse
    loss_te : float
        test rmse
    acc : float array
        accuracy (train and test)
    """
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
    """
    Cross validation for the gradient descent algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    k_indices : int array
        array of folded indices
    k : int
        index of the subset of index to form the test set
    max_iters : int
        number of iterations
    gamma : float
        step size

    Returns
    -------
    loss_tr : float
        training rmse
    loss_te : float
        test rmse
    acc : float array
        accuracy (train and test)
    """
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
    """
    Cross validation for the gradient descent algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    k_indices : int array
        array of folded indices
    k : int
        index of the subset of index to form the test set
    lambda_ : float
        regularisation parameter
    max_iters : int
        number of iterations
    gamma : float
        step size

    Returns
    -------
    loss_tr : float
        training rmse
    loss_te : float
        test rmse
    acc : float array
        accuracy (train and test)
    """
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


# def cross_validation_lambda_visualization(lambds, mse_tr, mse_te):
#     """visualization the curves of mse_tr and mse_te."""
#     plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
#     plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
#     plt.xlabel("lambda")
#     plt.ylabel("rmse")
#     plt.xlim(1e-4, 1)
#     plt.title("cross validation lambda")
#     plt.legend(loc=2)
#     plt.grid(True)
#     plt.savefig("cross_validation_lambda")


# def cross_validation_gamma_visualization(gams, mse_tr, mse_te):
#     """visualization the curves of mse_tr and mse_te."""
#     plt.semilogx(gams, mse_tr, marker=".", color='b', label='train error')
#     plt.semilogx(gams, mse_te, marker=".", color='r', label='test error')
#     plt.xlabel("lambda")
#     plt.ylabel("rmse")
#     #plt.xlim(1e-4, 1)
#     plt.title("cross validation gamma")
#     plt.legend(loc=2)
#     plt.grid(True)
#     plt.savefig("cross_validation_gamma")
    
def cv_polynomial_expansion(y, tx, k_indices, k, idx, degree):
    """
    Cross validation for the gradient descent algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    k_indices : int array
        array of folded indices
    k : int
        index of the subset of index to form the test set
    max_iters : int
        number of iterations
    idx : int
        index of the feature on which polynomial expansion is tested
    degree : int
        degree of polynomial expansion of the chosen feature

    Returns
    -------
    loss_tr : float
        training rmse
    loss_te : float
        test rmse
    acc_tr : float
        train accuracy
    acc_te : float
        test accuracy
    """
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    y_tr = np.delete(y, k_indices[k])
    tx_te = tx[k_indices[k]]
    y_te = y[k_indices[k]]
    
    new_tx_tr = build_poly_tx(tx_tr, idx, degree)
    new_tx_te = build_poly_tx(tx_te, idx, degree)
    
    weights, loss = least_squares(y_tr, new_tx_tr)
    loss_tr = np.sqrt(2 * compute_loss(y_tr, new_tx_tr, weights))
    loss_te = np.sqrt(2 * compute_loss(y_te, new_tx_te, weights))
    acc_tr, acc_te = accuracy(new_tx_tr, y_tr, new_tx_te, y_te, weights, print_= False)
              
    return loss_tr, loss_te, acc_tr, acc_te