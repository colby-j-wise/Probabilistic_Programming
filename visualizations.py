import pandas as pd
import numpy as np
import seaborn as sns
import data
import basis_functions
import matplotlib.pyplot as plt
import tensorflow as tf


def gp_reg_invert_K(x, y, x_star, kernel, kernel_params=[]):
    N = tf.cast(x.get_shape()[0], tf.int32).eval(session=tf.Session())
    K = kernel(x, x, *kernel_params)
    k_star = kernel(x_star, x, *kernel_params)
    k_star_star = kernel(x_star, x_star, *kernel_params)
    mu_n, sigma_n_sqr = tf.nn.moments(y, 1)
    K_noisy = K + sigma_n_sqr * tf.eye(N, dtype=tf.float64)
    K_noisy_inv = tf.matrix_inverse(K_noisy)
    f_bar = tf.matmul(tf.matmul(k_star, K_noisy_inv), y)
    tmp = tf.matmul(tf.matmul(k_star, K_noisy_inv), k_star, transpose_b=True)
    v = k_star_star - tmp

    k_y = kernel(y, y, *kernel_params)
    y_k_y = tf.matmul(tf.matmul(y, k_y, transpose_a=True), y)
    first = - 0.5 * y_k_y
    second = - 0.5 * tf.log(tf.norm(k_y))
    third = tf.cast(- tf.cast(N, dtype=tf.float32) /
                    2 * tf.log(2 * np.pi), dtype=tf.float64)
    logp = first + second + third

    return f_bar, v, logp


def visualize_by_borough(dataset):
    pickups = dataset.loc[:, ["pickup_longitude",
                              "pickup_latitude",
                              "pickup_neighborhood_name"]]
    pickups.columns = ["long", "lat", "neighborhood"]
    dropoffs = dataset.loc[:, ["dropoff_longitude",
                               "dropoff_latitude",
                               "dropoff_neighborhood_name"]]
    dropoffs.columns = ["long", "lat", "neighborhood"]
    data = pd.concat((pickups, dropoffs)).dropna()
    return sns.lmplot(x="long",
                      y="lat",
                      hue="neighborhood",
                      data=data,
                      fit_reg=False)


def vis_glm(num_pnts, indicator_cols, actual_data, qw, qb):
    times = np.linspace(0, 24, num_pnts)
    x_vis = pd.DataFrame({i: [0.0] * num_pnts for i in indicator_cols})
    x_vis["pickup_hour"] = times
    x_vis["pickup_timestamp"] = x_vis["pickup_hour"]\
        .apply(lambda x: pd.to_datetime(x, unit="s").hour)
    x_vis_stndzd = data.standardize_cols(x_vis).fillna(0.0)
    qw_i = qw.sample().eval()
    qb_i = qb.sample().eval()
    line = np.dot(x_vis_stndzd, qw_i) + qb_i
    plt.scatter(x=times, y=line)
    plt.scatter(x=actual_data["pickup_hour"], y=actual_data["trip_duration"])
    plt.xlabel("Pickup Hour")
    plt.ylabel("Trip Duration")
    plt.show()


def vis_glm_poly(num_pnts, degree, indicator_cols, actual_data, qw, qb):
    times = np.linspace(0, 24, num_pnts)
    x_vis = pd.DataFrame({i: [0.0] * num_pnts for i in indicator_cols})
    x_vis["pickup_hour"] = times
    x_vis["pickup_timestamp"] = x_vis["pickup_hour"]\
        .apply(lambda x: pd.to_datetime(x, unit="s").hour)
    x_vis_stndzd = basis_functions.polynomial(
        data.standardize_cols(x_vis).fillna(0.0), degree=degree)
    qw_i = qw.sample().eval()
    qb_i = qb.sample().eval()
    line = np.dot(x_vis_stndzd, qw_i) + qb_i
    plt.scatter(x=times, y=line)
    plt.scatter(x=actual_data["pickup_hour"], y=actual_data["trip_duration"])
    plt.xlabel("Pickup Hour")
    plt.ylabel("Trip Duration")
    plt.show()


def vis_gp(x, y, kernel, indicator_cols, num_samples, kernel_params=[],
           interp_pnts=100, title=""):
    times = np.linspace(0, 24, interp_pnts)
    x_vis = pd.DataFrame({i: [0.0] * interp_pnts for i in indicator_cols})
    x_vis["pickup_hour"] = times
    x_vis["pickup_timestamp"] = x_vis["pickup_hour"]\
        .apply(lambda x: pd.to_datetime(x, unit="s").hour)
    x_vis_stdzd = data.standardize_cols(x_vis).fillna(0.0)
    for i in range(num_samples):
        f, v, logp = gp_reg_invert_K(x, y, x_vis_stdzd, kernel,
                                     kernel_params=kernel_params)
        mean = np.reshape(f.eval(session=tf.Session()), -1)
        covariance = v.eval(session=tf.Session())
        samples = np.random.multivariate_normal(mean, covariance)
        plt.scatter(x=times, y=samples, marker=".", alpha=0.7, cmap=0.5)
    plt.xlabel("Pickup Hour")
    plt.ylabel("Trip Duration")
    plt.title(title)
    plt.show()
