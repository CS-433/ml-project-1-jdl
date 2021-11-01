import numpy as np


def standardize(x):
    """
    Standardize input set according to mean and std.

    Parameters
    ----------
    x : float array
        input data set

    Returns
    -------
    x : float array
        standardized set
    mean_x : float array
        means of each column (i.e. each feature)
    std_x : float array
        standard deviation of each column (i.e. each feature)
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(x_std, y_data):
    """
    Build model, to be used by ML methods.

    Parameters
    ----------
    x_std : float array
        input dataset
    y_data : binary array
        output of the input dataset

    Returns
    -------
    y : binary array
        output of the input dataset (unchanged)
    tx : float array
        input dataset modified to be used by ML methods
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

    Parameters
    ----------
    x : float array
        input dataset
    method : str
        method to be used for missing values replacement

    Returns
    -------
    x : float array
        dataset with replaced missing values
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
    Preprocessing of the train and test set to split the train set, replace missing values,
    standardize the dataset, and build the model.

    Parameters
    ----------
    x : float array
        train set
    y : binary array
        output of train set
    test_x : float array
        train set
    test_y : binary array
        output of train set
    method : str
        method to be used for missing values replacement
    ratio : float
        split ratio of the train set into a train and validation subset

    Returns
    -------
    train_tx : float array
        preprocessed train dataset
    train_y : binary array
        output of train dataset
    val_tx : float array
        preprocessed validation dataset
    val_y : binary array
        output of validation dataset
    test_tx : float array
        preprocessed test set
    test_y : binary array
        output of test set
    """
    # Split the training dataset into a training and validation set
    train_x, train_y, val_x, val_y = split_data(x, y, ratio, seed=1)

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
