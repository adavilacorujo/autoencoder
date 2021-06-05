import numpy as np
from activations import *

def foward_pass(X, W, B):
    Y = np.dot(X, W) + B
    return Y

def forward_prop(Layers, labels):
    x = Layers[0]['x']

    for i, layer in enumerate(Layers):
        w = layer['w']
        b = layer['b']
        actv = layer['activation']
        Y = foward_pass(x, w, b)

        if actv is not 'relu':
            x = mse(Y, labels)
        else:
            x = activation(Y, actv)
    return x

def add(layers : list, num_nodes : int, activation : str):
    w_prev_size = layers[-1]['w'].shape[1]
    w_current = np.random.randn(w_prev_size, num_nodes)
    layers.append({
        'b': 1,
        'w': w_current,
        'activation': activation
    })
    return layers

if __name__ == '__main__':
    layers = [
        {
            'x': np.random.randn(1, 3),
            'b': 1,
            'w': np.random.randn(3, 2),
            'activation': 'relu',
        },
    ]

    labels = np.random.randn(1, 3)
    layers = add(layers, 2, 'relu')
    layers = add(layers, 1, 'relu')
    layers = add(layers, 2, 'relu')
    layers = add(layers, 3, 'mse')

    output = forward_prop(layers, labels)
    print(labels)

    print(output)