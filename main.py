import preprocessing
import edward as ed
from edward.models import Normal
import tensorflow as tf
import numpy as np
import binning


def main():
    fname = "data/train.csv"
    clean_data = preprocessing.get_data(fname)
    # y = clean_data['trip_duration'].as_matrix()
    # drop fields that don't matter or we wouldn't know prior to the trip
    X = clean_data.drop(['id',
                         'vendor_id',
                         'pickup_datetime',
                         'dropoff_datetime',
                         'store_and_fwd_flag',
                         'trip_duration',
                         'dropoff_timestamp'],
                        axis=1).as_matrix()
    # bayesian_linear_regression(X, y)
    print(clean_data.columns)
    pickups = clean_data[:, ['pickup_longitude', 'pickup_latitude']]
    dropoffs = clean_data[:, ['pickup_longitude', 'pickup_latitude']]
    binning.bin_2d(np.concatenate((pickups, dropoffs)))


def bayesian_linear_regression(x_train, y_train):
    N, D = x_train.shape
    X = tf.placeholder(tf.float32, [N, D])
    w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
    b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
    y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

    # Inference
    qw = Normal(loc=tf.Variable(tf.random_normal([D])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
    qb = Normal(loc=tf.Variable(tf.random_normal([1])),
                scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

    inference = ed.KLpq({w: qw, b: qb}, data={X: x_train, y: y_train})
    inference.run(n_samples=5, n_iter=250)
    # TODO: model evaluation


if __name__ == '__main__':
    main()
