import pandas as pd
import math, quandl
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = 'wU5yTKUG-8e3aLv-8T_W'

df = quandl.get('WIKI/AMZN')
df = df[['Adj. Open', 'Adj. High', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"]
df['PCT_CHG'] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"]

df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'PCT_CHG']]
df.fillna(-999999, inplace = True)
forecast_column = "Adj. Close"

forecast_out = 150
print(forecast_out)
df['label'] = df[forecast_column].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace = True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = .2)

# clf = LinearRegression(n_jobs = -1)
# clf.fit(X_train, y_train)
# with open('linearregression.pickle', 'wb') as f:
# 	pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

clf.predict(X_train)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)


# print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()