import pandas as pd
import quandl


data_frame = quandl.get('WIKI/GOOGL')
data_frame = data_frame[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
data_frame['HL_PCT'] = (data_frame['Adj. High'] - data_frame['Adj. Low']) / data_frame['Adj. Low'] * 100
data_frame['PCT_change'] = (data_frame['Adj. Close'] - data_frame['Adj. Open']) / data_frame['Adj. Open'] * 100

data_frame = data_frame[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(data_frame.head())
