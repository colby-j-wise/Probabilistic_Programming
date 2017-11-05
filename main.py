import preprocessing


def main():
    fname = "data/train.csv"
    data = preprocessing.get_data(fname)
    print(data.describe())


def build_heatmap(dataset):
    pass


if __name__ == '__main__':
    main()
