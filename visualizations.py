import pandas as pd
import numpy as np
import seaborn as sns
import data
import basis_functions
import matplotlib.pyplot as plt
import edward as ed
import tensorflow as tf


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
    qw_i = tf.constant(qw.sample().eval(), tf.float64)
    qb_i = tf.constant(qb.sample().eval(), tf.float64)
    line = ed.models.Normal(loc=ed.dot(x_vis_stndzd, qw_i)
                            + qb_i, scale=tf.constant([1.0], tf.float64))\
        .sample().eval()
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
    qw_i = tf.constant(qw.sample().eval(), tf.float64)
    qb_i = tf.constant(qb.sample().eval(), tf.float64)
    line = ed.models.Normal(loc=ed.dot(x_vis_stndzd, qw_i)
                            + qb_i, scale=tf.constant([1.0], tf.float64))\
        .sample().eval()
    plt.scatter(x=times, y=line)
    plt.scatter(x=actual_data["pickup_hour"], y=actual_data["trip_duration"])
    plt.xlabel("Pickup Hour")
    plt.ylabel("Trip Duration")
    plt.show()
