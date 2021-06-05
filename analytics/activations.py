import numpy as np 


def RELU(Y):
    Y[ Y < 0 ] = 0
    return Y

def mse(Y, pred):
    ax = 0 # Perform along the row, for each column
    mse = (np.square(Y - pred)).mean(axis=ax)
    return mse

def mse_prime(Y, pred):
    return 2*(pred-Y)/len(Y);    

def rmse(Y, pred):
    ax = 0 
    rmse = np.sqrt(np.square(Y - pred)).mean(axis=ax)
    return rmse

activations = {
    "relu": RELU,
    "mse": mse,
    "rmse": rmse,
}


def activation(Y, activation):
    Z = activations[activation](Y)
    return Z