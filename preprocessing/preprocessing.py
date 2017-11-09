import numpy as np
import pandas as pd
import io
import preprocessing.neighborhoods
import time


def run():
    ROOT_DIR = "./preprocessing"
    INPUT_DIRECTORY = "/data"
    OUTPUT_DIRECTORY = "/output"
    N_DATA = ROOT_DIR + "/nyc_neighborhoods.json"
    shapes = preprocessing.neighborhoods.load_neighborhoods(N_DATA)
    print("starting time: " + time.asctime())
    try:
        with open(ROOT_DIR + OUTPUT_DIRECTORY + "./preprocessed.csv", "r") \
                as readable:
            dat = pd.read_csv(readable, header=None)
            last_id = dat.tail(1).iloc[0, 1]
            print("read last id from file where processing already began")
    except BaseException:
        last_id = None
    with open(ROOT_DIR + INPUT_DIRECTORY + "/train.csv", "r") as train:
        with open(ROOT_DIR + OUTPUT_DIRECTORY + "/preprocessed.csv", "a") \
                as processed:
            columns = train.readline()
            columns = columns[:len(columns) - 1].split(",")
            row_id = pd.read_csv(io.StringIO(train.readline()),
                                 header=None).iloc[0, 0]
            while last_id is not None and not last_id == row_id:
                row_id = \
                    pd.read_csv(io.StringIO(train.readline()),
                                header=None).iloc[0, 0]
            print("starting at id", row_id)
            for idx, line in enumerate(train):
                line_df = pd.read_csv(io.StringIO(line),
                                      names=columns)
                preprocessing.neighborhoods.add_neighborhoods(line_df,
                                                              shapes)
                s = io.StringIO()
                line_df.to_csv(s,
                               header=False)
                processed.write(s.getvalue())
                if idx % 5000 == 0:
                    print("reached row", idx)
                    print(s.getvalue())


def drop_outside_stddev(data, col, stddevs):
    return data[np.abs(data[col] - data[col].mean() <=
                       stddevs * data[col].std())].dropna()


def drop_less_than(data, col, less_than_val):
    return data[data[col] > less_than_val].dropna()


def drop_greater_than(data, col, greater_than_val):
    return data[data[col] < greater_than_val].dropna()


def datetime_to_epochmilli(date_col):
    return date_col.map(lambda x: x.timestamp())


def get_data(fname):
    d = pd.read_csv(fname,
                    parse_dates=['pickup_datetime',
                                 'dropoff_datetime'],
                    date_parser=pd.Timestamp)
    d = trim_lat_long_edges(d)
    d['pickup_timestamp'] = datetime_to_epochmilli(d['pickup_datetime'])
    d['dropoff_timestamp'] = datetime_to_epochmilli(d['dropoff_datetime'])
    return d


def trim_lat_long_edges(d):
    # cut out places west of the hudson river
    # 40.755701, -73.977154 - top right
    # 40.740783, -74.002973 - bottom left
    min_lat = 40.740783
    d = drop_less_than(d, 'pickup_latitude', min_lat)
    d = drop_less_than(d, 'dropoff_latitude', min_lat)
    # remove anything south of breezy poinnt
    min_long = -74.002973
    d = drop_less_than(d, 'pickup_longitude', min_long)
    d = drop_less_than(d, 'dropoff_longitude', min_long)
    # remove anything north of pelham bay
    max_lat = 40.755701
    d = drop_greater_than(d, 'pickup_latitude', max_lat)
    d = drop_greater_than(d, 'dropoff_latitude', max_lat)
    # remove things too far east
    max_long = -73.977154
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
