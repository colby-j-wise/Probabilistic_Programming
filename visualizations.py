import pandas as pd
import seaborn as sns


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
