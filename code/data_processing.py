import numpy as np


def standardize(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(x_std, y_data):
    y = y_data
    x = x_std
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def split_data(x, y, ratio, seed=1):
    # set seed
    np.random.seed(seed)

    # generate random indices
    data = np.vstack([y, x.T]).T
    per_data = np.random.permutation(data)
    idx = int(np.floor(x.shape[0] * ratio))
    train_data = per_data[:idx]
    test_data = per_data[idx:]
    train_x, train_y = train_data[:, 1:], train_data[:, 0]
    test_x, test_y = test_data[:, 1:], test_data[:, 0]

    return train_x, train_y, test_x, test_y
