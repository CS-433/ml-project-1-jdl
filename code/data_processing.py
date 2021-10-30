import numpy as np


def standardize(x):
    """
    Standardize input set according to mean and std.
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(x_std, y_data):
    """
    Build model, to be used by ML methods.
    """
    y = y_data
    x = x_std
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def split_data(x, y, ratio, seed=1):
    """
    Split the input dataset into two sub-datasets according to the ratio.

    Parameters
    ----------
    x : float array
        features set of the dataset to split
    y : binary array (0 or 1)
        prediction set of the dataset to split
    ratio : float
        ratio of the data split
    seed : int
        seed for the np.random

    Returns
    -------
    train_x : float array
        features set of the first sub-dataset (training)
    train_y : float array
        features set of the second sub-dataset (validation)
    test_x : float array
        prediction set of the first sub-dataset (training)
    test_y : float array
        prediction set of the second sub-dataset (validation)
    """
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


def replace_miss(x, method='mean'):
    """
    Replace missing values of each feature by the mean/median
    of the values of the feature.
    """
    if method == 'mean':
        # Replace -999 (=missing values) by NaN
        x[x == -999] = np.nan
        # Find the indices of NaN-values
        idx = np.where(np.isnan(x))
        # Calculate the column means of the remaining values
        means = np.nanmean(x, axis=0)
        # Replace the missing values by the corresponding mean
        x[idx] = np.take(means, idx[1])

    elif method == 'median':
        # Replace -999 (=missing values) by NaN
        x[x == -999] = np.nan
        # Find the indices of NaN-values
        idx = np.where(np.isnan(x))
        # Calculate the column medians of the remaining values
        medians = np.nanmedian(x, axis=0)
        # Replace the missing values by the corresponding median
        x[idx] = np.take(medians, idx[1])

    else:
        x = None  # or rise an error

    return x


def preprocessing(x, y, test_x, test_y, method='mean', ratio=0.7):
    """
    Preprocessing... (TO DO)
    """

    # Split the training dataset into a training and validation set
    train_x, train_y, val_x, val_y = split_data(x, y, ratio, seed=1)

    # replace all missing values by the coloum-mean of the remaining values in the training set
    # if dtype == 'mean':
    #     # replace -999 (=missing values) by nan
    #     train_x[train_x == -999] = np.nan
    #     val_x[val_x == -999] = np.nan
    #     test_x[test_x == -999] = np.nan
    #     # find in all three sets the indexes of nan-values
    #     train_idx = np.where(np.isnan(train_x))
    #     val_idx = np.where(np.isnan(val_x))
    #     test_idx = np.where(np.isnan(test_x))
    #     # calculate the coloumn means of the remaining values in the training set
    #     means = np.nanmean(train_x, axis=0)
    #     # replace the missing values by the corresponding mean
    #     train_x[train_idx] = np.take(means, train_idx[1])
    #     val_x[val_idx] = np.take(means, val_idx[1])
    #     test_x[test_idx] = np.take(means, test_idx[1])

        # replace all missing values by the coloum-median of the remaining values in the training set
    # elif dtype == 'median':
    #     # replace -999 (=missing values) by nan
    #     train_x[train_x == -999] = np.nan
    #     val_x[val_x == -999] = np.nan
    #     test_x[test_x == -999] = np.nan
    #     # find in all three sets the indexes of nan-values
    #     train_idx = np.where(np.isnan(train_x))
    #     val_idx = np.where(np.isnan(val_x))
    #     test_idx = np.where(np.isnan(test_x))
    #     # calculate the coloumn means of the remaining values in the training set
    #     medians = np.nanmedian(train_x, axis=0)
    #     # replace the missing values by the corresponding mean
    #     train_x[train_idx] = np.take(medians, train_idx[1])
    #     val_x[val_idx] = np.take(medians, val_idx[1])
    #     test_x[test_idx] = np.take(medians, test_idx[1])

        # deleting all features (=coloums) with missing values
    # elif dtype == 'col':
    #     # find in all three sets the coloums with at least one missing value
    #     train_idx = np.where(train_x == -999)[1]
    #     val_idx = np.where(val_x == -999)[1]
    #     test_idx = np.where(test_x == -999)[1]
    #     # list with all coloums that have at least one missing value in one of the three sets
    #     tot_idx = np.hstack([train_idx, val_idx, test_idx])
    #     # delete in all three sets those coloums
    #     train_x = np.delete(train_x, tot_idx, 1)
    #     val_x = np.delete(val_x, tot_idx, 1)
    #     test_x = np.delete(test_x, tot_idx, 1)

    # Replace missing values by the column mean
    train_x = replace_miss(train_x, method)
    val_x = replace_miss(val_x, method)
    test_x = replace_miss(test_x, method)

    # Standardize each feature in respect to the mean and std
    train_x, train_means, train_stds = standardize(train_x)
    val_x = (val_x - train_means) / train_stds
    test_x = (test_x - train_means) / train_stds

    # Build train, validation and test model (feature matrix tx, label vector y)
    train_y, train_tx = build_model_data(train_x, train_y)
    val_y, val_tx = build_model_data(val_x, val_y)
    test_y, test_tx = build_model_data(test_x, test_y)

    return train_tx, train_y, val_tx, val_y, test_tx, test_y
