import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split

def min_dist_arr(sample):
    """Find the minimum distance between two elements in an array"""
    return np.amin([ abs(x - y) for x, y in combinations(sample, 2) ])

def max_dist_arr(sample):
    """Find the maximum distance between two elements in an array"""
    return np.amax([ abs(x - y) for x, y in combinations(sample, 2) ])

def split_data(X_sample, Y_sample, test_size, seed):
    """Split data"""
    X_train, X_val, Y_train, Y_val = [
        x.copy()
    for x in train_test_split(X_sample, Y_sample, test_size=test_size, random_state=seed) ]

    return (X_train, X_val, Y_train, Y_val)

def gp_pred(gpr, X_val):
    """Predict values on validation data"""
    mu_val, std_val = gpr.predict(X_val, return_std=True)
    mu_val = [ x[0] for x in mu_val ]

    return (mu_val, std_val)

def gp_error(mu_val_train, mu_val_val, std_val_val, X_val, Y_train, Y_val):
    # Calculate relative difference between the target values and predicted values
    error_train = 100 * np.abs(np.divide(Y_train[Y_train.columns[0]] - mu_val_train, Y_train[Y_train.columns[0]]))
    error_val = 100 * np.abs(np.divide(Y_val[Y_val.columns[0]] - mu_val_val, Y_val[Y_val.columns[0]]))

    # Save prediction output results to a text file
    number_of_variables = len(X_val.columns)
    pred_outputs = np.empty((len(Y_val), 4+len(X_val.iloc[0,:])))
    pred_outputs[:,number_of_variables+0] = Y_val[Y_val.columns[0]]
    pred_outputs[:,number_of_variables+1] = mu_val_val
    pred_outputs[:,number_of_variables+2] = error_val
    pred_outputs[:,number_of_variables+3] = std_val_val
    pred_outputs[:,0:number_of_variables] = X_val
    header = "Variables_%d Target_value Predicted_value Error std" %number_of_variables
    np.savetxt("val_output.txt", pred_outputs, header=header)

    return (sum(error_train) / len(error_train), sum(error_val) / len(error_val))