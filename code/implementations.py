import numpy as np

from proj1_helpers import compute_gradient, compute_loss, \
    calculate_sigmoid_gradient, calculate_sigmoid_loss, batch_iter


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Implementation of the gradient descent algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    initial_w : float array
        initial weight vector
    max_iters : int
        number of iterations
    gamma : float
        step size

    Returns
    -------
    w : float array
        final weight vector
    loss : float
        loss value corresponding to the final weight vector
    """
    w = initial_w
    # Apply gradient descent over max_iters iteration
    for n_iter in range(max_iters):
        # Compute gradient
        gradient, _ = compute_gradient(y, tx, w)
        # Update w by gradient
        w = w - gamma * gradient

    # Compute loss of last w value
    loss = compute_loss(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Implementation of the stochastic gradient descent algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    initial_w : float array
        initial weight vector
    max_iters : int
        number of iterations
    gamma : float
        step size

    Returns
    -------
    w : float array
        final weight vector
    loss : float
        loss value corresponding to the final weight vector
    """
    w = initial_w
    batch_size = 1
    # Apply stochastic gradient descent over max_iters iteration
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute stochastic gradient
            stoch_gradient, _ = compute_gradient(minibatch_y, minibatch_tx, w)
            # Update w by stochastic gradient
            w = w - gamma * stoch_gradient

    # Compute loss of last w value
    loss = compute_loss(y, tx, w)

    return w, loss


# Least squares regression using normal equations
def least_squares(y, tx):
    """
    Implementation of the least squares algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset

    Returns
    -------
    w : float array
        final weight vector
    loss : float
        loss value corresponding to the final weight vector
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    # Compute w by solving the system
    w = np.linalg.solve(a, b)
    # Compute loss
    loss = compute_loss(y, tx, w)

    return w, loss


# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_=0.1):
    """
    Implementation of the ridge regression algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    lambda_ : float
        regularization parameter

    Returns
    -------
    w : float array
        final weight vector
    loss : float
        loss value corresponding to the final weight vector
    """
    a = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    # Compute w by solving the system
    w = np.linalg.solve(a, b)
    # Compute loss
    loss = compute_loss(y, tx, w)

    return w, loss


# Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Implementation of the logistic regression algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    initial_w : float array
        initial weight vector
    max_iters : int
        number of iterations
    gamma : float
        step size

    Returns
    -------
    w : float array
        final weight vector
    loss : float
        loss value corresponding to the final weight vector
    """
    w = initial_w
    # Apply gradient descent over max_iters iteration
    for n_iter in range(max_iters):
        # Compute gradient
        gradient = calculate_sigmoid_gradient(y, tx, w)
        # Update w by gradient
        w = w - gamma * gradient

    # Compute loss of last w value
    loss = calculate_sigmoid_loss(y, tx, w)
    
    return w, loss


# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, initial_w, lambda_, max_iters, gamma):
    """
    Implementation of the regularised logistic regression algorithm.

    Parameters
    ----------
    y : binary array
        output of the dataset
    tx : float array
        features dataset
    initial_w : float array
        initial weight vector
    lambda_ : float
        regularisation parameter
    max_iters : int
        number of iterations
    gamma : float
        step size

    Returns
    -------
    w : float array
        final weight vector
    loss : float
        loss value corresponding to the final weight vector
    """
    w = initial_w
    # Apply gradient descent over max_iters iteration
    for n_iter in range(max_iters):
        # Compute gradient
        gradient = calculate_sigmoid_gradient(y, tx, w) + 2*lambda_*w
        # Update w by gradient
        w = w - gamma * gradient

    # Compute loss of last w value
    loss = calculate_sigmoid_loss(y, tx, w) + lambda_ * np.linalg.norm(w, 2)**2

    return w, loss
