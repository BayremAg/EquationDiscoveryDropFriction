import numpy as np


def ReRMSE(y_pred, y_true):
    # relative root-mean-square error as in [Brence, Todorovski, and Džeroski, “Probabilistic Grammars for Equation
    # Discovery.” page 5]
    std = np.std(y_true)
    mse = np.mean(np.power(y_pred - y_true, 2) )
    rermse = np.sqrt(mse)/max(std,10e-18)

    return float(rermse)

def R2_y_true_mean_precomputed(args, X_df, y_pred):
    y_true = X_df.loc[:, 'y'].to_numpy()

    ss_res = np.sum((y_true - y_pred)**2)
    y_true_mean = X_df.loc[:, args.system_id_column].replace(args.liquid_surface_mean).astype(float).to_numpy()
    ss_tot = np.sum((y_true - y_true_mean)**2)
    r2 = 1 - (ss_res/ss_tot)
    return float(r2)



def Mse(y_pred, y_true):
    # Mean square error
    square_error = np.power(y_pred - y_true, 2)
    mse = np.mean(square_error)
    return float(mse)


def Me(y_pred, y_true):
    # Mean error
    me = np.mean(np.abs(y_true - y_pred))
    return float(me)
