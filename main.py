import numpy as np
import pandas as pd
import model as mp
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    start_time = datetime.now()

    cpi = pd.read_csv("CPIAUCSL.csv")
    m2 = pd.read_csv("M2SL.csv")
    data = cpi.merge(m2, how="right", left_on="DATE", right_on="DATE")
    data["CPIChange"] = (
        ((data["CPIAUCSL"] - data["CPIAUCSL"].shift(1) - 1) / (data["CPIAUCSL"].shift(1) - 1)).fillna(0))
    fed_funds = pd.read_csv("DFF.csv")
    data = data.merge(fed_funds, how="left", left_on="DATE", right_on="DATE")
    final_data = data.drop(["DATE"], axis=1)
    x_data = final_data.drop(["CPIAUCSL", "CPIChange"], axis=1)
    pd.set_option('display.max_rows', None)
    y_data = final_data["CPIChange"]
    x = final_data['M2SL'].to_numpy().reshape(final_data['M2SL'].to_numpy().shape[0], 1)
    y = final_data['CPIChange'].to_numpy().reshape(final_data['CPIChange'].to_numpy().shape[0], 1)

    b = np.random.rand(5, 1)
    b = np.append(1, b)
    x_vals = mp.normalize(x_data['M2SL'].to_numpy())

    x_all = mp.nonlinear_transform(x_vals.reshape(x_vals.shape[0], 1), 5)

    x_all = np.concatenate((np.ones((x.shape[0], 1)), x_all), axis=1)

    y_vals = mp.normalize(y_data.to_numpy())

    y_vals = y_vals.reshape(y_vals.shape[0], 1)

    test_weights, losses = mp.nonlinear_gradient_descent(b, x_all, y_vals, 0.3, 30000)

    test_y = [mp.f(test_weights, val) for val in x_all]

    plt.figure()
    plt.scatter(x_vals, y_vals)
    plt.plot(x_vals, test_y, c="r")
    plt.title("Gradient Descent")
    plt.xlabel("M2 Money Supply")
    plt.ylabel("Change in CPI")
    plt.show()
    plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    end_time = datetime.now()
    print('Gradient Descent Duration: {}'.format(end_time - start_time))


    start_time = datetime.now()

    b2 = np.random.rand(5)
    x_vals = mp.normalize(x_data['M2SL'].to_numpy())
    x_all = mp.nonlinear_transform(x_vals.reshape(x_vals.shape[0], 1), 5)
    y_vals = mp.normalize(y_data.to_numpy())
    y_vals = y_vals.reshape(y_vals.shape[0], 1)
    weights = mp.gauss_newton(b2, x_all, y_vals)
    test_y = [mp.f(weights, val) for val in x_all]

    plt.figure()
    plt.scatter(x_all[:, 0], y_vals)
    plt.plot(x_vals, test_y, c="r")
    plt.title("Gauss-Newton")
    plt.xlabel("M2 Money Supply")
    plt.ylabel("Change in CPI")
    plt.show()

    end_time = datetime.now()
    print('Gauss-Newton Duration: {}'.format(end_time - start_time))

    start_time = datetime.now()

    x_vals = mp.normalize(x_data['M2SL'].to_numpy())
    y_vals = mp.normalize(y_data.to_numpy())

    betas = np.random.rand(5, 1)
    betas = np.append(betas, 1)

    popt = mp.lm(mp.function, x_vals, y_vals, betas)

    y_test = mp.function(x_vals, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])

    plt.figure()
    plt.scatter(x_vals, y_vals)
    plt.plot(x_vals, y_test, c="r")
    plt.title("Levenberg-Marquardt")
    plt.xlabel("M2 Money Supply")
    plt.ylabel("Change in CPI")
    plt.show()

    end_time = datetime.now()
    print('Levenberg-Marquardt Duration: {}'.format(end_time - start_time))
