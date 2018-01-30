import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


data_set = pd.read_csv('../data_sets/breast-cancer-wisconsin.txt')
data_set.replace('?', -99999, inplace=True)
data_set.drop('id', 1, inplace=True)

X = np.array(data_set.drop('class', 1))
y = np.array(data_set['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = svm.SVC()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = classifier.predict(example_measures)

print(accuracy)
print(prediction)
