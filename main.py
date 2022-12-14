import numpy as np
import pandas as pd
import model as mp
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import f

if __name__ == '__main__':
    #Set the start time and print the start time
    np.set_printoptions(threshold=np.inf)
    start_time = datetime.now()
    print(start_time)

    #read in and manipulate the data as needed
    cpi = pd.read_csv("CPIAUCSL.csv")
    m2 = pd.read_csv("M2SL.csv")
    data = cpi.merge(m2, how="right", left_on="DATE", right_on="DATE")
    data["CPIChange"] = (
        ((data["CPIAUCSL"] - data["CPIAUCSL"].shift(1) - 1) / (data["CPIAUCSL"].shift(1) - 1)).fillna(0))
    fed_funds = pd.read_csv("DFF.csv")
    data = data.merge(fed_funds, how="left", left_on="DATE", right_on="DATE")

    final_data = data.drop(["DATE"], axis=1)
    shuffle_df = final_data.sample(frac=1)
    train_size = int(0.7 * len(final_data))
    train_set = shuffle_df[:train_size]

    x_data = train_set.drop(["CPIAUCSL", "CPIChange"], axis=1)
    pd.set_option('display.max_rows', None)
    y_data = train_set["CPIChange"]
    x = train_set['M2SL'].to_numpy().reshape(train_set['M2SL'].to_numpy().shape[0], 1)
    y = train_set['CPIChange'].to_numpy().reshape(train_set['CPIChange'].to_numpy().shape[0], 1)

    #set the weights and the inputs for the Gradient Descent algorithm
    b = np.random.rand(5, 1)
    b = np.append(1, b)
    x_vals = mp.normalize(x_data['M2SL'].to_numpy())
    x_all = mp.nonlinear_transform(x_vals.reshape(x_vals.shape[0], 1), 5)
    x_all = np.concatenate((np.ones((x.shape[0], 1)), x_all), axis=1)
    y_vals = mp.normalize(y_data.to_numpy())
    y_vals = y_vals.reshape(y_vals.shape[0], 1)
    test_weights, losses = mp.nonlinear_gradient_descent(b, x_all, y_vals, 0.75, 500000)
    test_y = [mp.f(test_weights, val) for val in x_all]
    print("Gradient Descent Cost")
    print(mp.loss_mse(y_vals, test_y))
    plt.figure()
    plt.scatter(x_vals, y_vals)
    plt.scatter(x_vals, test_y, c="r")
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

    # set the weights and the inputs for the Gauss-Newton algorithm
    b2 = np.random.rand(5)
    x_vals = mp.normalize(x_data['M2SL'].to_numpy())
    x_all = mp.nonlinear_transform(x_vals.reshape(x_vals.shape[0], 1), 5)
    y_vals = mp.normalize(y_data.to_numpy())
    y_vals = y_vals.reshape(y_vals.shape[0], 1)
    weights = mp.gauss_newton(b2, x_all, y_vals)
    test_y = [mp.f(weights, val) for val in x_all]
    print("Gauss-Newton Cost")
    print(mp.loss_mse(y_vals, test_y))
    plt.figure()
    plt.scatter(x_all[:, 0], y_vals)
    plt.scatter(x_vals, test_y, c="r")
    plt.title("Gauss-Newton")
    plt.xlabel("M2 Money Supply")
    plt.ylabel("Change in CPI")
    plt.show()

    end_time = datetime.now()
    print('Gauss-Newton Duration: {}'.format(end_time - start_time))

    start_time = datetime.now()

    # set the weights and the inputs for the Levenberg-Marquardt algorithm
    x_vals = mp.normalize(x_data['M2SL'].to_numpy())
    y_vals = mp.normalize(y_data.to_numpy())
    betas = np.random.rand(5, 1)
    betas = np.append(betas, 1)
    popt = mp.lm(mp.function, x_vals, y_vals, betas)
    y_test = mp.function(x_vals, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    print("Levenberg-Marquardt Cost")
    print(mp.loss_mse(y_vals, y_test))
    plt.figure()
    plt.scatter(x_vals, y_vals)
    plt.scatter(x_vals, y_test, c="r")
    plt.title("Levenberg-Marquardt")
    plt.xlabel("M2 Money Supply")
    plt.ylabel("Change in CPI")
    plt.show()

    end_time = datetime.now()
    print('Levenberg-Marquardt Duration: {}'.format(end_time - start_time))

    #get the test data
    test_set = shuffle_df[train_size:]
    x_data_test = test_set.drop(["CPIAUCSL", "CPIChange"], axis=1)
    pd.set_option('display.max_rows', None)

    #set the weights and calculate the test estimates for the Gradient Descent algorithm
    b = np.random.rand(5, 1)
    b = np.append(1, b)
    x_vals_test = mp.normalize(x_data_test['M2SL'].to_numpy())
    x_all_test_GD = mp.nonlinear_transform(x_vals_test.reshape(x_vals_test.shape[0], 1), 5)
    x_all_test_GD = np.concatenate((np.ones((x_all_test_GD.shape[0], 1)), x_all_test_GD), axis=1)

    test_GD = [mp.f(test_weights, val) for val in x_all_test_GD]
    test_GD = mp.normalize(np.array(test_GD))
    print("Test Gradient Descent Cost")
    print(mp.loss_mse(test_set["CPIChange"].to_numpy(), np.array(test_GD)))

    plt.figure()
    plt.scatter(mp.normalize(x_data_test['M2SL'].to_numpy()), mp.normalize(test_set["CPIChange"].to_numpy()))
    plt.scatter(mp.normalize(x_data_test['M2SL'].to_numpy()), np.array(test_GD), c="r")
    plt.title("Gradient Descent Test")
    plt.xlabel("M2 Money Supply")
    plt.ylabel("Change in CPI")
    plt.show()

    # set the weights and calculate the test estimates for the Gauss-Newton algorithm
    b2 = np.random.rand(5)
    b2 = np.append(b2, 1)
    x_vals_test_GN = mp.normalize(x_data_test['M2SL'].to_numpy())
    x_all_test_GN = mp.nonlinear_transform(x_vals_test_GN.reshape(x_vals_test.shape[0], 1), 5)

    test_GN = [mp.f(weights, val) for val in x_all_test_GN]
    test_GN = mp.normalize(np.array(test_GN))
    print("Test Gauss-Newton Cost")
    print(mp.loss_mse(test_set["CPIChange"].to_numpy(), np.array(test_GN)))

    plt.figure()
    plt.scatter(mp.normalize(x_data_test['M2SL'].to_numpy()), mp.normalize(test_set["CPIChange"].to_numpy()))
    plt.scatter(mp.normalize(x_data_test['M2SL'].to_numpy()), np.array(test_GN), c="r")
    plt.title("Gauss-Newton Test")
    plt.xlabel("M2 Money Supply")
    plt.ylabel("Change in CPI")
    plt.show()

    # set the weights and calculate the test estimates for the Levenberg-Marquardt algorithm
    x_vals_test_LM = mp.normalize(x_data_test['M2SL'].to_numpy())
    betas = np.random.rand(5, 1)
    betas = np.append(betas, 1)
    test_LM = mp.function(x_vals_test_LM, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    test_LM = mp.normalize(np.array(test_LM))
    print("Test Levenberg-Marquardt Cost")
    print(mp.loss_mse(test_set["CPIChange"].to_numpy(), np.array(test_LM)))

    plt.figure()
    plt.scatter(mp.normalize(x_data_test['M2SL'].to_numpy()), mp.normalize(test_set["CPIChange"].to_numpy()))
    plt.scatter(mp.normalize(x_data_test['M2SL'].to_numpy()), np.array(test_LM), c="r")
    plt.title("Levenberg-Marquardt Test")
    plt.xlabel("M2 Money Supply")
    plt.ylabel("Change in CPI")
    plt.show()

    # f-scores for the algorithms
    print("Gauss-Newton vs Gradient Descent")
    F = mp.f_stat(mp.normalize(test_set["CPIChange"].to_numpy()), test_GN,
                       mp.normalize(test_set["CPIChange"].to_numpy()), test_GD)
    p_value = 1 - f.cdf(F, len(mp.normalize(test_set["CPIChange"].to_numpy()) - 1),
                        len(mp.normalize(test_set["CPIChange"].to_numpy())) - 1)
    print(p_value)

    print("Gradient Descent vs Levenberg-Marquardt")
    F = mp.f_stat(mp.normalize(test_set["CPIChange"].to_numpy()), test_GD,
                  mp.normalize(test_set["CPIChange"].to_numpy()), test_LM)
    p_value = 1 - f.cdf(F, len(mp.normalize(test_set["CPIChange"].to_numpy())) - 1,
                        len(mp.normalize(test_set["CPIChange"].to_numpy())) - 1)
    print(p_value)

    print("Gauss-Newton vs Levenberg-Marquardt")
    F = mp.f_stat(mp.normalize(test_set["CPIChange"].to_numpy()), test_GN,
                       mp.normalize(test_set["CPIChange"].to_numpy()), test_LM)
    p_value = 1 - f.cdf(F, len(mp.normalize(test_set["CPIChange"].to_numpy()) - 1),
                        len(mp.normalize(test_set["CPIChange"].to_numpy()) - 1))
    print(p_value)
