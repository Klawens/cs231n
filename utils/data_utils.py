import pickle
import os
import numpy as np


def load_CIFAR10_batch(filename):
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
        X = dict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(dict['labels'])

        return X, Y


def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR10_batch(os.path.join(ROOT, 'test_batch'))

    return Xtr, Ytr, Xte, Yte
