import numpy as np
import pandas as pd
import model as mp
import matplotlib.pyplot as plt


GDweights = np.array([0.21092408,5.86132156,-19.21112451,20.62260688,-1.15165511,-5.56927788])


GNweights = np.array([12.8911585,-74.0697061,179.21413897,-190.68538024,73.63833142])


LMweights = np.array([49.39970994,-129.00835852,123.06085002,-52.13425684,9.47686819,0.14767845])

cpi = pd.read_csv("CPIAUCSL.csv")
m2 = pd.read_csv("M2SL.csv")
data = cpi.merge(m2, how="right", left_on="DATE", right_on="DATE")
data["CPIChange"] = (((data["CPIAUCSL"] - data["CPIAUCSL"].shift(1) - 1) / (data["CPIAUCSL"].shift(1) - 1)).fillna(0))
data["CPIChange"] = mp.normalize(data["CPIChange"])

data["NormM2"] = mp.normalize(data["M2SL"])

x_data = data[data["DATE"] > "2022-10-01"]["NormM2"]

x_all = mp.nonlinear_transform(x_data.to_numpy().reshape(x_data.shape[0], 1), 5)

x_GD = np.concatenate((np.ones((x_data.shape[0], 1)), x_all), axis=1)

infl = data[data["DATE"] > "2022-10-01"]["CPIChange"]

test_GD = [mp.f(GDweights, val) for val in x_GD]

test_GN = [mp.f(GNweights, val) for val in x_all]

test_LM = [mp.f(LMweights, val) for val in x_GD]

plt.figure()
plt.scatter(x_data, infl)
plt.plot(x_data, np.array(test_GD), c="r")
plt.title("Gradient Descent")
plt.xlabel("M2 Money Supply")
plt.ylabel("Change in CPI")
plt.show()

plt.figure()
plt.scatter(x_data, infl)
plt.plot(x_data, np.array(test_GN), c="r")
plt.title("Gradient Descent")
plt.xlabel("M2 Money Supply")
plt.ylabel("Change in CPI")
plt.show()

plt.figure()
plt.scatter(x_data, infl)
plt.plot(x_data, np.array(test_LM), c="r")
plt.title("Gradient Descent")
plt.xlabel("M2 Money Supply")
plt.ylabel("Change in CPI")
plt.show()