import numpy as np


def ReRMSE(y_pred, y_true):
    # relative root-mean-square error as in [Brence, Todorovski, and Džeroski, “Probabilistic Grammars for Equation
    # Discovery.” page 5]
    std = np.std(y_true)
    mse = np.mean(np.power(y_pred - y_true, 2) )
    rermse = np.sqrt(mse)/max(std,10e-18)

    return float(rermse)



def Mse(y_pred, y_true):
    # Mean square error
    square_error = np.power(y_pred - y_true, 2)
    mse = np.mean(square_error)
    return float(mse)


def Me(y_pred, y_true):
    # Mean error
    me = np.mean(np.abs(y_true - y_pred))
    return float(me)
