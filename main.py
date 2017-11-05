import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import preprocessing


def main():
    fname = "data/train.csv"
    data = preprocessing.get_data(fname)
    j = data[['pickup_longitude', 'pickup_latitude']].as_matrix()
    k = data[['dropoff_longitude', 'dropoff_latitude']].as_matrix()
    z = np.concatenate((j, k))
    sns.jointplot(x=z[:, 0], y=z[:, 1])
    plt.show()


def build_heatmap(dataset):
    pass


if __name__ == '__main__':
    main()
