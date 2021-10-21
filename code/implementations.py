import numpy as np


# Functions for the calculation of the loss
def calculate_mse(e):
    return 1 / 2 * np.mean(e ** 2)


def calculate_mae(e):
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)


# Compute gradient and error vector
def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    n = len(y)

    gradient = -1 / n * tx.T.dot(e)
    return gradient, e


# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    # Apply gradient descent over max_iters iteration
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient, _ = compute_gradient(y, tx, w)
        # Update w by gradient
        w = w - gamma * gradient

    # Compute loss of last w value
    loss = compute_loss(y, tx, w)
    return w, loss


# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        # Pick mini-batch
        data_size = len(y)
        sample_index = np.random.randint(0, data_size)
        minibatch_y = y[sample_index:sample_index+1]
        minibatch_tx = tx[sample_index:sample_index+1, :]
        # Compute SG and loss
        stoch_gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        # Update w by stochastic gradient
        w = w - gamma * stoch_gradient

    # Compute loss of last w value
    loss = compute_loss(y, tx, w)
    return w, loss


# Least squares regression using normal equations
def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_=0.1):
    A = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    x = tx.T.dot(y)
    w = np.linalg.solve(A, x)
    loss = compute_loss(y, tx, w)
    return w, loss


# Logistic regression using gradient descent or SGD
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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    # Apply gradient descent over max_iters iteration
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = calculate_sigmoid_gradient(y, tx, w)
        # Update w by gradient
        w = w - gamma * gradient

    # Compute loss of last w value
    loss = calculate_sigmoid_loss(y, tx, w)
    
    return w, loss


# Regularized logistic regression using gradient descent or SGD
def help_penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_sigmoid_loss(y, tx, w) + lambda_ * np.linalg.norm(w, 2)**2
    gradient = calculate_sigmoid_gradient(y, tx, w) + 2*lambda_*w
    hessian = calculate_sigmoid_hessian(y, tx, w) + 2*lambda_
    return loss, gradient, hessian

def reg_logistic_regression(y, tx, initial_w, lambda_, max_iters, gamma):
    w = initial_w
    # Apply gradient descent over max_iters iteration
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient = calculate_sigmoid_gradient(y, tx, w) + 2*lambda_*w
        # Update w by gradient
        w = w - gamma * gradient

    # Compute loss of last w value
    loss = calculate_sigmoid_loss(y, tx, w) + lambda_ * np.linalg.norm(w, 2)**2
    
    return w, loss