import numpy as np
import pandas as pd
import io
import neighborhoods
import time


def run():
    INPUT_DIRECTORY = "data"
    OUTPUT_DIRECTORY = "output"
    N_DATA = "./nyc_neighborhoods.json"
    neighborhood_shapes = neighborhoods.load_neighborhoods(N_DATA)
    print("starting time: " + time.asctime())
    try:
        with open(OUTPUT_DIRECTORY + "/xpreprocessed.csv", "r") as readable:
            num_lines_already_written = \
                [i for i, j in enumerate(readable)][-1] + 1
    except OSError:
        num_lines_already_written = 1
    with open(INPUT_DIRECTORY + "/xtrain.csv", "r") as train:
        with open(OUTPUT_DIRECTORY + "/xpreprocessed.csv", "a") as processed:
            columns = train.readline()
            columns = columns[:len(columns) - 1].split(",")
            for idx, line in enumerate(train):
                if idx < num_lines_already_written:
                    continue
                line_df = pd.read_csv(io.StringIO(line),
                                      names=columns)
                neighborhoods.add_neighborhoods(line_df, neighborhood_shapes)
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
