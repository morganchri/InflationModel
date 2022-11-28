import numpy as np
import pandas as pd
import model as mp

if __name__ == '__main__':
    # x = 2 * np.random.rand(100, 1)
    # y = 4 + 3 * x + np.random.randn(100, 1)
    # theta = np.random.randn(2, 1)
    # x_b = np.c_[np.ones((len(x), 1)), x]
    # theta, c_hist, t_hist = mp.linear_gradient_descent(x_b, y, theta)
    # print(theta[0][0])
    # print(theta[1][0])

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

    weight_3, bias_3, train_losses_3 = mp.polynomial_gradient_descent(
        mp.normalize(final_data['M2SL'].to_numpy().reshape(final_data['M2SL'].to_numpy().shape[0], 1)),
        mp.normalize(final_data['CPIChange'].to_numpy().reshape(final_data['CPIChange'].to_numpy().shape[0], 1)),
        200, 0.01, 3)
    print(weight_3)
    print(bias_3)
