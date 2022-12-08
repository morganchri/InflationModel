import numpy as np
import pandas as pd
import model as mp
import matplotlib.pyplot as plt
from datetime import datetime


if __name__ == '__main__':

    start_time = datetime.now()

    cpi = pd.read_csv("CPIAUCSL.csv")
    m2 = pd.read_csv("M2SL.csv")
    data = cpi.merge(m2, how="right", left_on="DATE", right_on="DATE")
    data["CPIChange"] = (
        ((data["CPIAUCSL"] - data["CPIAUCSL"].shift(1) - 1) / (data["CPIAUCSL"].shift(1) - 1)).fillna(0))
    fed_funds = pd.read_csv("DFF.csv")
    data = data.merge(fed_funds, how="left", left_on="DATE", right_on="DATE")
    final_data = data.drop(["DATE"], axis=1)
    x_data = final_data.drop(["CPIAUCSL", "CPIChange", "DFF"], axis=1)
    pd.set_option('display.max_rows', None)
    y_data = final_data["CPIChange"]
    x = final_data['M2SL'].to_numpy().reshape(final_data['M2SL'].to_numpy().shape[0], 1)
    y = final_data['CPIChange'].to_numpy().reshape(final_data['CPIChange'].to_numpy().shape[0], 1)
    b = np.random.rand(5, 1)
    b = np.append(b, 1)
    x_vals = mp.normalize(x_data['M2SL'].to_numpy())
    y_vals = mp.normalize(y_data.to_numpy())

    test_weights = mp.nonlinear_gradient_descent(b, x_vals, y_vals, 0.1, 20000)

    test_y = [mp.f(test_weights, val) for val in x_vals]

    plt.scatter(x_vals, y_vals)
    plt.plot(x_vals, test_y, c="r")
    plt.show()

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
