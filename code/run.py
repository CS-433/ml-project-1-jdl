import zipfile

from implementations import least_squares
from data_processing import preprocessing
from proj1_helpers import *
from cross_validation import build_k_indices, cv_polynomial_expansion

"""Load train and test data"""
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
zf = zipfile.ZipFile('../data/test.csv.zip')
zf.extract('test.csv', path='../data')
DATA_TEST_PATH = '../data/test.csv'
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

"""Feature processing"""
# Remove non-used features
removed_indices_missing = [4, 5, 6, 12, 26, 27, 28]
removed_indices_univariate = [15, 18, 20, 25]
removed_indices_covariate = [9, 29]
removed_indices = removed_indices_univariate + removed_indices_covariate + removed_indices_missing

tX = np.delete(tX, removed_indices, 1)
tX_test = np.delete(tX_test, removed_indices, 1)

# Feature preprocessing (split, missing values replacement, standardisation and model building)
train_tx, train_y, val_tx, val_y, test_tx, test_y = preprocessing(tX, y, tX_test, y_test, method='mean', ratio=0.7)

"""Polynomial feature expansion (using cross validation)"""
# Cross validation parameter setting
seed = 1
k_fold = 5
degrees = np.arange(1, 11)
features = np.arange(1, len(train_tx[0]))

# Split data in k fold
k_indices = build_k_indices(train_y, k_fold, seed)

# Define lists to store the loss/accuracy of training data and test data
rmse_tr = np.zeros((len(train_tx[0]),len(degrees)))
rmse_te = np.zeros((len(train_tx[0]),len(degrees)))
acc_tr = np.zeros((len(train_tx[0]),len(degrees)))
acc_te = np.zeros((len(train_tx[0]),len(degrees)))

# Compute loss/accuracy for each degree and each feature
for i in features:
    for j, degree in enumerate(degrees):
        rmse_tr_tmp = []
        rmse_te_tmp = []
        acc_tr_tmp = []
        acc_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te, a_tr, a_te = cv_polynomial_expansion(train_y, train_tx, k_indices, k, [i], degree)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
            acc_tr_tmp.append(a_tr)
            acc_te_tmp.append(a_te)
        rmse_tr[i, j] = np.mean(rmse_tr_tmp)
        rmse_te[i, j] = np.mean(rmse_te_tmp)
        acc_tr[i, j] = np.mean(acc_tr_tmp)
        acc_te[i, j] = np.mean(acc_te_tmp)

best_degrees = []
for i in features:
    best_degrees.append(np.argmax(acc_te[i]))

"""Compute expanded datasets"""
expanded_train_tx = build_final_poly_tx(train_tx, best_degrees)
expanded_val_tx = build_final_poly_tx(val_tx, best_degrees)
expanded_test_tx = build_final_poly_tx(test_tx, best_degrees)

"""Compute model"""
expanded_weights, expanded_loss = least_squares(train_y, expanded_train_tx)

"""Write in output file"""
OUTPUT_PATH = '../data/sample-submission.csv'
y_pred = predict_labels(expanded_weights, expanded_test_tx)
y_pred[y_pred == 0] = -1
print(y_pred.shape)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
