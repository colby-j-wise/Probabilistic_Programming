import preprocessing.preprocessing
import edward as ed
from edward.models import Normal
import tensorflow as tf
import numpy as np


def main():
    fname = "preprocessing/data/xtrain.csv"
    clean_data = preprocessing.preprocessing.get_data(fname)
    y = clean_data['trip_duration'].as_matrix()
    # drop fields that don't matter or we wouldn't know prior to the trip
    X = clean_data.drop(['id',
                         'vendor_id',
                         'pickup_datetime',
                         'dropoff_datetime',
                         'store_and_fwd_flag',
                         'trip_duration',
                         'dropoff_timestamp'],
                        axis=1).as_matrix()
    # split to train/test
    num_train_samples = int(0.9 * X.shape[0])
    train_indices = np.random.choice(num_train_samples,
                                     size=num_train_samples,
                                     replace=False)
    x_train = X[train_indices, :]
    y_train = y[train_indices]
    x_test = np.delete(X, train_indices, axis=0)
    y_test = np.delete(y, train_indices)
    bayesian_lin_reg(x_train, y_train, x_test, y_test)


def bayesian_lin_reg(x_train, y_train, x_test, y_test):
    N, D = x_train.shape
    X = tf.placeholder(tf.float32, [N, D])
    rbf = tf.map_fn(lambda x: tf.reshape(ed.rbf(tf.reshape(x, (D, 1))), [D * D]), X)
    w = Normal(loc=tf.zeros(D * D), scale=tf.ones(D * D))
    b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
    y = Normal(loc=ed.dot(rbf, w) + b, scale=tf.ones(N))

    qw = Normal(loc=tf.Variable(tf.random_normal([D * D])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([D * D]))))
    qb = Normal(loc=tf.Variable(tf.random_normal([1])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
    inference = ed.KLpq({w: qw, b: qb}, data={X: x_train, y: y_train})
    inference.run(n_samples=50, n_iter=1000)
    posterior_predictive = Normal(loc=ed.dot(X, qw), scale=tf.ones(N))
    ed.evaluate('mean_squared_error',
                data={X: x_test,
                      posterior_predictive: y_test})


def eval(x_test, y_test):
    pass


if __name__ == '__main__':
    main()
