import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

cpi = pd.read_csv("CPIAUCSL.csv")
m2 = pd.read_csv("M2SL.csv")
data = cpi.merge(m2, how="right", left_on="DATE", right_on="DATE")
data["CPIChange"] = (((data["CPIAUCSL"] - data["CPIAUCSL"].shift(1) - 1) / (data["CPIAUCSL"].shift(1) - 1)).fillna(0))
fed_funds = pd.read_csv("DFF.csv")
data = data.merge(fed_funds, how="left", left_on="DATE", right_on="DATE")


# plt.scatter(data["M2SL"], data["CPIChange"])
# plt.title("M2")
# plt.show()
# plt.scatter(data["DFF"], data["CPIChange"])
# plt.title("Fed Funds")
# plt.show()

