import numpy as np


def metric(y_true, y_pred):

    factor = np.ones(y_true.shape)
    factor[y_pred<0.5] = 1.2
    se = np.abs(y_true - y_pred)**2
    mse = np.mean(factor * se)
    return np.sqrt(mse)