import numpy as np


def load_data(dataset_type='train'):
    dataset_type = dataset_type.lower().strip()
    if dataset_type == 'train':
        path_dataset = "../data/train.csv"
    elif dataset_type == 'test':
        # Path to adapt because no in the repo
        path_dataset = "../data/test.csv"
    feature_cols = np.arange(32)[2:]
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=feature_cols)
    features = data
    labels = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1], dtype=bytes,
        converters={1: lambda x: 0 if b"s" in x else 1})
    return features, labels

