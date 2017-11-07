import shapely.geometry
import json
import numpy as np


def load_neighborhoods(path):
    with open(path, "r") as data:
        raw_json = json.load(data)
    neighborhoods_list = []
    for neighborhood in raw_json["features"]:
        border = shapely.geometry.asShape(neighborhood["geometry"])
        name = neighborhood["properties"]["neighborhood"]
        borough = neighborhood["properties"]["borough"]
        neighborhoods_list.append({"shape": border,
                                   "neighborhood": name,
                                   "borough": borough})
    return neighborhoods_list


def add_neighborhoods(data, neighborhoods_list):
    pickup, borough = get_pickup_neighborhood(data, neighborhoods_list)
    data["pickup_neighborhood"] = pickup
    data["pickup_borough"] = borough
    dropoff, borough = get_dropoff_neighborhood(data, neighborhoods_list)
    data["dropoff_neighborhood"] = dropoff
    data["dropoff_borough"] = borough
    return data


def get_pickup_neighborhood(row, neighborhoods):
    longitude = row["pickup_longitude"]
    latitude = row["pickup_latitude"]
    return get_neighborhood((longitude, latitude), neighborhoods)


def get_dropoff_neighborhood(row, neighborhoods):
    longitude = row["dropoff_longitude"]
    latitude = row["dropoff_latitude"]
    return get_neighborhood((longitude, latitude), neighborhoods)


def get_neighborhood(point, neighborhoods):
    for neighborhood in neighborhoods:
        if neighborhood["shape"].contains(shapely.geometry.Point(point[0],
                                                                 point[1])):
            return neighborhood["neighborhood"], neighborhood["borough"]
    return np.nan, np.nan
