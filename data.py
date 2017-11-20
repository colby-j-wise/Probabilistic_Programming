import pandas as pd
import numpy as np
from preprocessing.preprocessing import get_data


def get_borough_data(path, borough):
    full_dataset = get_data(path)
    return full_dataset.where((full_dataset["pickup_boro"] == borough) &
                              (full_dataset["dropoff_boro"] == borough))\
                       .dropna()


def get_neighborhood_to_neighborhood(source_neighborhood, sink_neighborhood,
                                     full_dataset):
    return full_dataset.where((full_dataset["pickup_neighborhood_name"] ==
                               source_neighborhood) &
                              (full_dataset["dropoff_neighborhood_name"] ==
                               sink_neighborhood)).dropna()


def clean_data(data):
    data = __remove_outliers(data, "trip_duration", 1.5)
    data = __remove_leq_zero(data, "trip_duration")
    return data.reset_index(drop=True)


def add_handy_columns(data):
    d1 = __add_manhattan_distance(data)
    d2 = __add_pickup_hour(d1)
    d3 = __add_dropoff_hour(d2)
    d4 = __add_pickup_day_of_week(d3)
    d5 = __add_dropoff_day_of_week(d4)
    return d5


def __add_pickup_day_of_week(x):
    z = x.copy()
    z["pickup_day_of_week"] = x["pickup_datetime"].apply(lambda x: x.dayofweek)
    return z


def __add_dropoff_day_of_week(x):
    if "dropoff_datetime" not in x.columns:
        x = __add_dropoff_datetime(x)
    z = x.copy()
    z["dropoff_day_of_week"] = x["dropoff_datetime"].apply(lambda x:
                                                           x.dayofweek)
    return z


def __add_dropoff_timestamp(x):
    z = x.copy()
    z["dropoff_timestamp"] = x["pickup_timestamp"] + x["trip_duration"]
    return z


def __add_dropoff_datetime(x):
    z = x.copy()
    z["dropoff_datetime"] = x["pickup_datetime"] + \
        x["trip_duration"].apply(lambda x: pd.Timedelta(seconds=x))
    return z


def __add_dropoff_hour(data):
    if "dropoff_datetime" not in data.columns:
        data = __add_dropoff_datetime(data)
    z = data.copy()
    z["dropoff_hour"] = data["dropoff_datetime"].apply(lambda x: x.hour)
    return z


def __add_pickup_hour(x):
    z = x.copy()
    z["pickup_hour"] = x["pickup_datetime"].apply(lambda x: x.hour)
    return z


def __add_manhattan_distance(data):
    z = data.copy()
    z["manhattan_distance"] = abs(data["pickup_longitude"] -
                                  data["dropoff_longitude"] +
                                  data["pickup_latitude"] -
                                  data["dropoff_latitude"])
    return z


def __remove_outliers(data, col, frac_stddev):
    bound = frac_stddev * data.describe()[col]["std"]
    return data.where(np.abs(data[col]) < bound).dropna()


def __remove_leq_zero(data, col):
    return data.where(data[col] > 0).dropna()
