import pandas as pd


def polynomial(data, degree):
    assert isinstance(data, pd.DataFrame)
    ret = data.copy()
    for i in range(degree):
        for col in ret:
            ret[col + "_poly_degree_" + str(i)] = ret[col] ** i
    return ret
