import numpy as np


def ReMSe(y_pred, y_true):
    # relative root-mean-square error as in [Brence, Todorovski, and Džeroski, “Probabilistic Grammars for Equation
    # Discovery.”]
    mean_y_true = np.mean(y_true)
    mean_square_error = Mse(y_pred, y_true)
    variance = np.mean(np.power(y_true - mean_y_true, 2))
    remse = np.divide(np.sqrt(mean_square_error), variance + 10e-10)
    return float(remse)


def Mse(y_pred, y_true):
    # Mean square error
    square_error = np.power(y_pred - y_true, 2)
    mse = np.mean(square_error)
    return float(mse)


def Me(y_pred, y_true):
    # Mean error
    me = np.mean(np.abs(y_true - y_pred))
    return float(me)
