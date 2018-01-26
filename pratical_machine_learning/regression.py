import pandas as pd
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

print(data_frame.head())
