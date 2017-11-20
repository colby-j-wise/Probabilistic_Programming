import pandas as pd


def polynomial(data, degree):
    if degree == 1:
        return data.copy()
    assert isinstance(data, pd.DataFrame)
    ret = data.copy()
    for col in ret:
        for i in range(2, degree + 1):
            ret[col + "_poly_degree_" + str(i)] = ret[col] ** i
    return ret
