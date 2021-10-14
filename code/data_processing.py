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


def preprocessing(features, labels, col=True, row=False, ratio=0.7):
    # deleting all features (=coloums) with missing values
    if col:
        idx = np.where(features == -999)[1]
        processed_f = np.delete(features, idx, 1)
        processed_l = labels

    # deleting all rows with missing values
    elif row:
        idx = np.where(features == -999)[0]
        processed_f = np.delete(features, idx, 0)
        processed_l = np.delete(labels, idx, 0)

    # standardize each feature
    processed_f, mean_f, std_f = standardize(processed_f)

    # split the data into a training and test set
    train_x, train_y, test_x, test_y = split_data(processed_f, processed_l, ratio, seed=1)

    # build train- and testmodel (feature matrix tx, label vector y)
    train_y, train_tx = build_model_data(train_x, train_y)
    test_y, test_tx = build_model_data(test_x, test_y)

    return train_tx, train_y, test_tx, test_y


def correctness(train_tx, train_y, test_tx, test_y, weights):
    # Make predictions
    train_pred = train_tx.dot(weights)
    test_pred = test_tx.dot(weights)

    # Transform the prediction into 0 ( = 's') and 1 (= 'b')
    train_pred = np.where(train_pred > 0.5, 1, 0)
    test_pred = np.where(test_pred > 0.5, 1, 0)

    # Compute the ratio of correct labled predictions
    train_score = np.sum(np.where(train_pred == train_y, 1, 0)) / len(train_pred)
    test_score = np.sum(np.where(test_pred == test_y, 1, 0)) / len(test_pred)

    print("There are {train_s}% correct prediction in the training set".format(train_s=train_score * 100))
    print("There are {test_s}% correct prediction in the test set".format(test_s=test_score * 100))

    return train_score, test_score