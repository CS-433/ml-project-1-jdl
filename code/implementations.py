import numpy as np

from helpers import compute_gradient, compute_loss, \
    calculate_sigmoid_gradient, calculate_sigmoid_loss


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
