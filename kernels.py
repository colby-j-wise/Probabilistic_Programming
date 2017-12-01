import tensorflow as tf
import numpy as np
import scipy.spatial as spt


# Needto make sure the data subset was PSD
# since tf.cholesky requires postitive semi definite matrix
def is_positive_definite(X):
	#X = X.eval()
	if ( np.all(np.linalg.eigvals(X) > 0) ):
		return True
	else:
		return False


def RationalQuadratic(X, X2=None, lengthScale=1.0, alpha=1.0): 
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
    square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - 2 * tf.matmul(X, X2, transpose_b=True)
    output =  1 + (square / 2)
    K = tf.pow(output, -alpha)
    return K





