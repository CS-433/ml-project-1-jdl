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


def preprocessing(x, y, test_x, test_y, dtype = 'mean', ratio = 0.7):
    
    #split the training data into a training and validation set
    train_x, train_y, val_x, val_y = split_data(x, y, ratio, seed=1)
    
    #replace all missing values by the coloum-mean of the remaining values in the training set
    if dtype == 'mean':
        #replace -999 (=missing values) by nan
        train_x[train_x == -999] = np.nan
        val_x[val_x == -999] = np.nan
        test_x[test_x == -999] = np.nan
        #find in all three sets the indexes of nan-values
        train_idx = np.where(np.isnan(train_x))
        val_idx = np.where(np.isnan(val_x))
        test_idx = np.where(np.isnan(test_x))
        #calculate the coloumn means of the remaining values in the training set
        means = np.nanmean(train_x, axis = 0)
        #replace the missing values by the corresponding mean        
        train_x[train_idx] = np.take(means, train_idx[1]) 
        val_x[val_idx] = np.take(means, val_idx[1])
        test_x[test_idx] = np.take(means, test_idx[1]) 

        
    #replace all missing values by the coloum-median of the remaining values in the training set   
    elif dtype == 'median':
        #replace -999 (=missing values) by nan
        train_x[train_x == -999] = np.nan
        val_x[val_x == -999] = np.nan
        test_x[test_x == -999] = np.nan
        #find in all three sets the indexes of nan-values
        train_idx = np.where(np.isnan(train_x))
        val_idx = np.where(np.isnan(val_x))
        test_idx = np.where(np.isnan(test_x))
        #calculate the coloumn means of the remaining values in the training set
        medians = np.nanmedian(train_x, axis = 0)
        #replace the missing values by the corresponding mean        
        train_x[train_idx] = np.take(medians, train_idx[1]) 
        val_x[val_idx] = np.take(medians, val_idx[1])
        test_x[test_idx] = np.take(medians, test_idx[1]) 
        
    #deleting all features (=coloums) with missing values
    elif dtype == 'col':
        #find in all three sets the coloums with at least one missing value
        train_idx = np.where(train_x == -999)[1]
        val_idx = np.where(val_x == -999)[1]
        test_idx = np.where(test_x == -999)[1]
        #list with all coloums that have at least one missing value in one of the three sets
        tot_idx = np.hstack([train_idx, val_idx, test_idx])
        #delete in all three sets those coloums
        train_x = np.delete(train_x, tot_idx, 1)
        val_x = np.delete(val_x, tot_idx, 1)
        test_x = np.delete(test_x, tot_idx, 1)
    
    #deleting all rows with missing values. Not possible for submission predictions, since we have to predict each row!
    elif dtype == 'row':
        #find in all three sets the rows with at least one missing value
        train_idx = np.where(train_x == -999)[0]
        val_idx = np.where(val_x == -999)[0]
        test_idx = np.where(test_x == -999)[0]
        #delete in all three sets those rows (in the x matrix and the y vector)
        train_x = np.delete(train_x, train_idx, 0)
        train_y = np.delete(train_y, train_idx, 0)
        val_x = np.delete(val_x, val_idx, 0)
        val_y = np.delete(val_y, val_idx, 0)
        test_x = np.delete(test_x, test_idx, 0)
        test_y = np.delete(test_y, test_idx, 0)
        
    #standardize each feature in respect to the mean and std in the training set
    train_x, train_means, train_stds = standardize(train_x)
    val_x = (val_x - train_means) / train_stds
    test_x = (test_x - train_means) / train_stds
    
    #build train-, validation and testmodel (feature matrix tx, label vector y)
    train_y, train_tx = build_model_data(train_x, train_y)
    val_y, val_tx = build_model_data(val_x, val_y)
    test_y, test_tx = build_model_data(test_x, test_y) 
           
    return train_tx, train_y, val_tx, val_y, test_tx, test_y


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