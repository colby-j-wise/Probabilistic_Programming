import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform



def rbf(X, X2=None, lengthscale=1.0, variance=1.0):
    lengthscale = tf.convert_to_tensor(lengthscale, tf.float64)
    variance = tf.convert_to_tensor(variance, tf.float64)

    X = tf.convert_to_tensor(X)
    X = X / lengthscale
    Xs = tf.reduce_sum(tf.square(X), 1)
    if X2 is None:
        X2 = X
        X2s = Xs
    else:
        X2 = tf.convert_to_tensor(X2)
        X2 = X2 / lengthscale
        X2s = tf.reduce_sum(tf.square(X2), 1)

    square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - \
        2 * tf.matmul(X, X2, transpose_b=True)
    output = variance * tf.exp(-square / 2)
    return output


def RationalQuadratic(X, X2=None, lengthScale=0.5, alpha=0.1):
    lengthScale = tf.convert_to_tensor(lengthScale, tf.float64)
    alpha = tf.convert_to_tensor(alpha, tf.float64)
    X = tf.convert_to_tensor(X)
    X = X / alpha
    X = X / lengthScale
    Xs = tf.reduce_sum(tf.square(X), 1)
    if X2 is None:
        X2 = X
        X2s = Xs
    else:
        X2 = tf.convert_to_tensor(X2)
        X2 = X2 / alpha
        X2 = X2 / lengthScale
        X2s = tf.reduce_sum(tf.square(X2), 1)
    square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - 2 *\
        tf.matmul(X, X2, transpose_b=True)
    output =  1 + (square / 2)
    K = tf.pow(output, -alpha)
    return K


def ExpSineSquared(X, Xs=None, lengthScale=0.5, period=8.0):
    if Xs is None:
        X = X.eval(session=tf.Session())
        print
        dists = squareform(pdist(X, metric='euclidean'))
        arg = np.pi * dists / period
        sin_arg = np.sin(arg)
        K = np.exp(- 2 * (sin_arg / lengthScale) ** 2)
    else:
        X = X.eval(session=tf.Session())
        Xs = Xs.eval(session=tf.Session())
        dists = cdist(X, Xs, metric='euclidean')
        K = np.exp(- 2 * (np.sin(np.pi / period * dists) 
                    / lengthScale) ** 2)
    return tf.convert_to_tensor(K, tf.float64)
