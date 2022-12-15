import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

cpi = pd.read_csv("CPIAUCSL.csv")
m2 = pd.read_csv("M2SL.csv")
data = cpi.merge(m2, how="right", left_on="DATE", right_on="DATE")
data["CPIChange"] = (((data["CPIAUCSL"] - data["CPIAUCSL"].shift(1) - 1) / (data["CPIAUCSL"].shift(1) - 1)).fillna(0))
fed_funds = pd.read_csv("DFF.csv")
data = data.merge(fed_funds, how="left", left_on="DATE", right_on="DATE")
print(data.shape[0])

plt.figure()
plt.scatter(data["M2SL"], data["CPIChange"])
plt.title("M2")
plt.xlabel("M2 Money Supply")
plt.ylabel("Inflation")
plt.show()
plt.figure()
plt.scatter(data["DFF"], data["CPIChange"])
plt.title("Fed Funds")
plt.xlabel("Federal Funds Rate")
plt.ylabel("Inflation")
plt.show()
plt.figure()
plt.scatter(data["M2SL"], data["DFF"])
plt.title("Correlation Between M2 and Fed Funds")
plt.xlabel("M2 Money Supply")
plt.ylabel("Federal Funds Rate")
plt.show()
# print(data)
# print(data.drop(['CPIAUCSL', 'DATE'], axis=1))


pd.set_option('display.max_rows', None)
tax = pd.read_csv("IITTRHB.csv")
tax["DATE"] = pd.to_datetime(tax["DATE"])
tax = tax.set_index("DATE").resample("M").last().bfill().reset_index()
tax['DATE'] = tax['DATE'].dt.strftime('%Y-%m-01')

data = data.merge(tax, how="left", left_on="DATE", right_on="DATE")
data = data.dropna()

plt.figure()
plt.scatter(data["IITTRHB"], data["CPIChange"])
plt.title("Correlation Between Taxes and Inflation")
plt.xlabel("Tax Rate")
plt.ylabel("Inflation")
plt.show()

print("M2 and DFF")
print(stats.spearmanr(data.drop(['CPIAUCSL', 'DATE'], axis=1)["M2SL"],
                      data.drop(['CPIAUCSL', 'DATE'], axis=1)["DFF"]))
print("M2 and Taxes")
print(stats.spearmanr(data.drop(['CPIAUCSL', 'DATE'], axis=1)["M2SL"],
                      data.drop(['CPIAUCSL', 'DATE'], axis=1)["IITTRHB"]))
print("DFF and Taxes")
print(stats.spearmanr(data.drop(['CPIAUCSL', 'DATE'], axis=1)["DFF"],
                      data.drop(['CPIAUCSL', 'DATE'], axis=1)["IITTRHB"]))
print("M2 and Inflation")
print(stats.spearmanr(data.drop(['CPIAUCSL', 'DATE'], axis=1)["M2SL"],
                      data.drop(['CPIAUCSL', 'DATE'], axis=1)["CPIChange"]))
print("DFF and Inflation")
print(stats.spearmanr(data.drop(['CPIAUCSL', 'DATE'], axis=1)["DFF"],
                      data.drop(['CPIAUCSL', 'DATE'], axis=1)["CPIChange"]))
print("Taxes and Inflation")
print(stats.spearmanr(data.drop(['CPIAUCSL', 'DATE'], axis=1)["IITTRHB"],
                      data.drop(['CPIAUCSL', 'DATE'], axis=1)["CPIChange"]))

gdp = pd.read_csv("GDP.csv")
gdp["DATE"] = pd.to_datetime(gdp["DATE"])
gdp = gdp.set_index("DATE").resample("M").last().bfill().reset_index()
gdp['DATE'] = gdp['DATE'].dt.strftime('%Y-%m-01')



#print(gdp)
