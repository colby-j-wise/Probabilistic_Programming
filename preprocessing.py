import numpy as np
import pandas as pd


def drop_outside_stddev(data, col, stddevs):
    return data[np.abs(data[col] - data[col].mean() <=
                       stddevs * data[col].std())].dropna()


def drop_less_than(data, col, less_than_val):
    return data[data[col] > less_than_val].dropna()


def drop_greater_than(data, col, greater_than_val):
    return data[data[col] < greater_than_val].dropna()


def get_data(fname):
    d = pd.read_csv(fname,
                    parse_dates=['pickup_datetime',
                                 'dropoff_datetime'],
                    date_parser=pd.to_datetime)
    # cut out places west of the hudson river
    min_lat = 40.538
    d = drop_less_than(d, 'pickup_latitude', min_lat)
    d = drop_less_than(d, 'dropoff_latitude', min_lat)
    # remove anything south of breezy poinnt
    min_long = -74.032607
    d = drop_less_than(d, 'pickup_longitude', min_long)
    d = drop_less_than(d, 'dropoff_longitude', min_long)
    # remove anything north of pelham bay
    max_lat = 41.0
    d = drop_greater_than(d, 'pickup_latitude', max_lat)
    d = drop_greater_than(d, 'dropoff_latitude', max_lat)
    # remove things too far east
    max_long = -73.807216
    d = drop_greater_than(d, 'pickup_longitude', max_long)
    d = drop_greater_than(d, 'pickup_longitude', max_long)

    # remove anything trips than one minute in duration
    d = drop_less_than(d, 'trip_duration', 60)
    # remove anything over five hours in duration
    d = drop_greater_than(d, 'trip_duration', 60 * 60 * 5)
    # remove anything outside 2 std devs
    d = drop_outside_stddev(d, 'trip_duration', 2)
    # remove anywhere that the pickup was 2 std devs out
    d = drop_outside_stddev(d, 'pickup_longitude', 2)
    d = drop_outside_stddev(d, 'pickup_latitude', 2)
    # remove anywhere that the dropoff was 2 std devs out
    d = drop_outside_stddev(d, 'dropoff_longitude', 2)
    d = drop_outside_stddev(d, 'dropoff_latitude', 2)

    return d
