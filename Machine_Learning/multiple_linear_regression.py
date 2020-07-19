from matplotlib import pyplot as plt
from sklearn import linear_model
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("FuelConsumption.csv")
cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY",
          "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB",
          "CO2EMISSIONS"]]

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Multiple Regression Model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[["ENGINESIZE", "CYLINDERS",
                         "FUELCONSUMPTION_CITY",
                         "FUELCONSUMPTION_HWY"]])
y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(x, y)

# Prediction
y_hat = regr.predict(test[["ENGINESIZE", "CYLINDERS",
                           "FUELCONSUMPTION_CITY",
                           "FUELCONSUMPTION_HWY"]])
x = np.asanyarray(test[["ENGINESIZE", "CYLINDERS",
                       "FUELCONSUMPTION_CITY",
                        "FUELCONSUMPTION_HWY"]])
y = np.asanyarray(test[["CO2EMISSIONS"]])

print("Coefficients: ", regr.coef_)
print("Residul sum of squares: %.2f" % np.mean((y_hat - y) ** 2))
print("Variance score: %.2f" % regr.score(x, y))


