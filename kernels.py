import tensorflow as tf
import numpy as np
import scipy.spatial as spt

# Kernel Math Equations @: http://docs.pymc.io/api/gp.html#pymc3.gp.cov.Constant

# Needto make sure the data subset was PSD
# since tf.cholesky requires postitive semi definite matrix
def is_positive_definite(X):
	#X = X.eval()
	if ( np.all(np.linalg.eigvals(X) > 0) ):
		return True
	else:
		return False


def Kernel32Matern(X, X2=None, lengthScale=1.0):

	#K(x,x') = (1 + ( (x - x')^2/2al^2) )^-a
	lengthScale = tf.convert_to_tensor(lengthScale)
	#dependencies = [tf.assert_positive(lengthScale)]
	#lengthScale = control_flow_ops.with_dependencies(dependencies, lengthScale)

	X = tf.convert_to_tensor(X)
	X = X / lengthScale
	Xs = tf.reduce_sum(tf.square(X), 1) 
	if X2 is None:
		X2 = X
		X2s = Xs
	else:
		X2 = tf.convert_to_tensor(X2)
		X2 = X2 / lengthScale
		X2s = tf.reduce_sum(tf.square(X2), 1)

	square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - 2 * tf.matmul(X, X2, transpose_b=True)
	sqrt = tf.sqrt(3*square)
	output1 =  1 + square 
	power = tf.pow(output1, -alpha)
	output2 = matern_exp(X, lengthScale)
	K = tf.matmul(output1, output2, transpose_b=False)
	return K

def matern_exp(X, Xs=None, lengthScale=1.0):
	lengthScale = tf.convert_to_tensor(lengthScale)

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

	square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - 2 * tf.matmul(X, X2, transpose_b=True)
	sqrt = tf.sqrt(3*square)
	output = tf.exp(-square)
	return output

def RationalQuadratic(X, X2=None, lengthScale=1.0, alpha=1.0): #Not sure if good defaults

	#K(x,x') = (1 + ( (x - x')^2/2al^2) )^-a
	lengthScale = tf.convert_to_tensor(lengthScale)
	alpha = tf.convert_to_tensor(alpha)
	#dependencies = [tf.assert_positive(lengthScale), tf.assert_positive(alpha)]

	#lengthScale = tf.cond(dependencies, lengthScale)
	#alpha = control_flow_ops.with_dependencies(dependencies, alpha)

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

	if (is_positive_definite(K.eval(session=sess))):
		print("PSD")
	else:
		print("Not PSD")
	
	return K

def PolyKernel(X, degree=3):
	K = ((tf.matmul(X, X, transpose_b=True)) + 1.0)**degree
	return tf.convert_to_tensor(K) 


# From Edward
def rbf(X, X2=None, lengthscale=1.0, variance=1.0):
  """Radial basis function kernel, also known as the squared
  exponential or exponentiated quadratic. It is defined as
  $k(x, x') = \sigma^2 \exp\Big(
      -\\frac{1}{2} \sum_{d=1}^D \\frac{1}{\ell_d^2} (x_d - x'_d)^2 \Big)$
  for output variance $\sigma^2$ and lengthscale $\ell^2$.
  The kernel is evaluated over all pairs of rows, `k(X[i, ], X2[j, ])`.
  If `X2` is not specified, then it evaluates over all pairs
  of rows in `X`, `k(X[i, ], X[j, ])`. The output is a matrix
  where each entry (i, j) is the kernel over the ith and jth rows.
  Args:
    X: tf.Tensor.
      N x D matrix of N data points each with D features.
    X2: tf.Tensor, optional.
      N x D matrix of N data points each with D features.
    lengthscale: tf.Tensor, optional.
      Lengthscale parameter, a positive scalar or D-dimensional vector.
    variance: tf.Tensor, optional.
      Output variance parameter, a positive scalar.
  #### Examples
  ```python
  X = tf.random_normal([100, 5])
  K = ed.rbf(X)
  assert K.shape == (100, 100)
  ```
  """
  lengthscale = tf.convert_to_tensor(lengthscale)
  variance = tf.convert_to_tensor(variance)
  dependencies = [tf.assert_positive(lengthscale),
                  tf.assert_positive(variance)]
  lengthscale = control_flow_ops.with_dependencies(dependencies, lengthscale)
  variance = control_flow_ops.with_dependencies(dependencies, variance)

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

  square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - 2 * tf.matmul(X, X2, transpose_b=True)
  output = variance * tf.exp(-square / 2)
  return output
