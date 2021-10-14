import numpy as np

from implementations import (least_squares_GD, least_squares_SGD,
                             least_squares, ridge_regression,
                             logistic_regression, reg_logistic_regression)
from load_data import load_data
from data_processing import preprocessing, correctness, standardize, \
    build_model_data


'''Build model using x method'''
# Load data
features, labels = load_data()
# Preprocessing
train_tx, train_y, test_tx, test_y = preprocessing(features, labels)
# Model computation using least_squares
weights, loss = least_squares(train_y, train_tx)

'''Make prediction from test.csv'''
# Load test features
test_features, test_prediction = load_data('test')
# Extract features (manually or reuse preprocessing with ratio=1)
#idx = np.where(features == -999)[1]
#processed_f = np.delete(test_features, idx, 1)
#processed_l = test_prediction
#processed_f, _, _ = standardize(processed_f)
#_, test_tx = build_model_data(processed_f, processed_l)
test_tx, _, _, _ = preprocessing(test_features, test_prediction, ratio=1)
# Compute prediction
pred = test_tx.dot(weights)
# Convert predictions in 's' or 'b'
idx_b = pred >= 0.5
idx_s = pred < 0.5
pred = pred.astype(dtype='|S10')  # Note: dtype='|S10' == output.dtype and allows to replace int by str
pred[idx_b] = 'b'
pred[idx_s] = 's'

'''Write predictions in csv file'''
# Define path of template and output files
template_file_path = '../data/sample-submission.csv'
output_file_path = '../data/test_prediction_least_squares.csv'
# Insert predictions in template
output = np.genfromtxt(template_file_path, dtype=str, delimiter=',', skip_header=0)
output[1:, 1] = pred
# Save output file
np.savetxt(output_file_path, output, delimiter=",", fmt='%s')