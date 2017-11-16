from preprocessing.preprocessing import get_data


def get_borough_data(path, borough):
    full_dataset = get_data(path)
    return full_dataset.where((full_dataset["pickup_boro"] == borough) &
                              (full_dataset["dropoff_boro"] == borough))\
                       .dropna()
