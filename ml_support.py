import numpy as np
import pandas
# from ggplot import *
# import scipy
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn import linear_model

"""
Functions to calculate R2 and normalize featured for machine learning. 
"""

def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.
    
    y_bar = data.mean()
    res_ss = np.square(data - predictions).sum()
    var = np.square(data - y_bar).sum()
    r_squared = 1 - res_ss/var

    return r_squared
