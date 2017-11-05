import preprocessing
import edward as ed
from edward.models import Normal
import tensorflow as tf


def main():
    fname = "data/train.csv"
    data = preprocessing.get_data(fname)
    y = data['trip_duration'].as_matrix()
    X = data.drop(['id',
                   'vendor_id',
                   'pickup_datetime',
                   'dropoff_datetime',
                   'store_and_fwd_flag',
                   'trip_duration',
                   'dropoff_timestamp'],
                  axis=1).as_matrix()
    bayesian_linear_regression(X, y)


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


if __name__ == '__main__':
    main()
