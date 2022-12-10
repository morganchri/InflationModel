import numpy as np
import pandas as pd
import model as mp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def function(x, a, b, c, d, e, f):
    return (a * x ** 5) + (b * x ** 4) + (c * x ** 3) + (d * x ** 2) + (e * x) + f


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
b = np.random.rand(5, 1)
b = np.append(b, 1)
x_vals = mp.normalize(x_data['M2SL'].to_numpy())
y_vals = mp.normalize(y_data.to_numpy())

popt, pcov = curve_fit(function, x_vals, y_vals, p0=b)

y_test = function(x_vals, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])

plt.figure()
plt.scatter(x_vals, y_vals)
plt.plot(x_vals, y_test, c="r")
plt.show()
