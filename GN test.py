import numpy as np
import model as mp
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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
b = np.random.rand(5)
x_vals = mp.normalize(x_data['M2SL'].to_numpy())
x_all = mp.nonlinear_transform(x_vals.reshape(x_vals.shape[0], 1), 5)
y_vals = mp.normalize(y_data.to_numpy())
y_vals = y_vals.reshape(y_vals.shape[0], 1)
weights = mp.gauss_newton(b, x_all, y_vals)
test_y = [mp.f(weights, val) for val in x_all]

plt.figure()
plt.scatter(x_all[:, 0], y_vals)
plt.plot(x_vals, test_y, c="r")
plt.show()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
