import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing

style.use('ggplot')


def handle_non_numerical_data(data_frame):
    columns = data_frame.columns.values

    for column in columns:
        text_digit_values = {}

        if data_frame[column].dtype != np.int64 and data_frame[column].dtype != np.float64:
            x = 0
            column_contents = data_frame[column].values.tolist()
            unique_elements = set(column_contents)

            for element in unique_elements:
                if element not in text_digit_values:
                    text_digit_values[element] = x
                    x += 1

            data_frame[column] = list(map(lambda val: text_digit_values[val], data_frame[column]))

    return data_frame


data_frame = pd.read_excel('../data_sets/titanic.xls')
data_frame.drop(['name', 'body', 'boat', 'sex'], 1, inplace=True)
data_frame.fillna(0, inplace=True)

data_frame = handle_non_numerical_data(data_frame)

X = np.array(data_frame.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(data_frame['survived'].astype(float))

classifier = KMeans(n_clusters=2)
classifier.fit(X)

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i])
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = classifier.predict(predict_me)

    if prediction == y[i]:
        correct += 1

print(correct / len(X))
