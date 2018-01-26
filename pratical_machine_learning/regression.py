from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import quandl
import math

data_frame = quandl.get('WIKI/GOOGL')
data_frame = data_frame[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
data_frame['HL_PCT'] = (data_frame['Adj. High'] - data_frame['Adj. Low']) / data_frame['Adj. Low'] * 100
data_frame['PCT_change'] = (data_frame['Adj. Close'] - data_frame['Adj. Open']) / data_frame['Adj. Open'] * 100

data_frame = data_frame[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_column = 'Adj. Close'
data_frame.fillna(-99999, inplace=True)
forecast_output = int(math.ceil(0.01 * len(data_frame)))

data_frame['label'] = data_frame[forecast_column].shift(-forecast_output)
data_frame.dropna(inplace=True)

X = np.array(data_frame.drop(['label'], 1))
y = np.array(data_frame['label'])
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

classifier = LinearRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

print(accuracy)
