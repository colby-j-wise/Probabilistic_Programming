import pandas as pd
import tensorflow as tf
import numpy as np


def get_data(path, name):
    """
    path - str - relative path to data
    returns - tf.constant - the longitude (x axis) and latitude (y axis)
                            coordinates of all pickups and dropoffs in the
                            dataset
    """
    data = pd.read_csv(path,
                       parse_dates=['pickup_datetime',
                                    'dropoff_datetime'],
                       date_parser=pd.to_datetime)
    pickup_coordinates = data.loc[:, ['pickup_longitude',
                                      'pickup_latitude']].as_matrix()
    dropoff_coordinates = data.loc[:, ['dropoff_longitude',
                                       'dropoff_latitude']].as_matrix()
    coords = np.concatenate((pickup_coordinates, dropoff_coordinates))
    return tf.constant(coords, name=name, dtype=tf.float32)


def rotate(data, theta):
    """
    data - a 2-dimensional tensor of shape (?, 2)
    theta - the degree of rotation
    """
    rotation_matrix = tf.convert_to_tensor([[tf.cos(theta), -tf.sin(theta)],
                                            [tf.sin(theta), tf.cos(theta)]])
    return tf.matmul(rotation_matrix, data, transpose_b=True)


def bin_2d(dataset, num_bins):
    """
    dataset is a two dimensional numpy array of shape (?, 2)
    """
    start_longitude = np.min(dataset[:, 0])
    stop_longitude = np.max(dataset[:, 0])
    start_latitude = np.min(dataset[:, 1])
    stop_latitude = np.max(dataset[:, 1])

    step_size_longitude = (stop_longitude - start_longitude) / num_bins
    step_size_latitude = (stop_latitude - start_latitude) / num_bins

    xedges = np.arange(start_longitude, stop_longitude, step_size_longitude)
    yedges = np.arange(start_latitude, stop_latitude, step_size_latitude)
    H = np.histogram2d(dataset[:, 0],
                       dataset[:, 1],
                       bins=(xedges, yedges))
    return H[0]
