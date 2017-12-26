import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

data_set = pandas.read_csv(url, names=names)


# Quantity of rows and columns
# print(data_set.shape)

# Peek the first x rows
# print(data_set.head(20))

# Statical summary
# print(data_set.describe())

# Class distribution
# print(data_set.groupby('class').size())

# Data visualization (Univariate Plots)
# Box and whisker plots
# data_set.plot(kind='box', subplots=True, layout=(2,2), sharex=False)
# plt.show()

# Histograms
# data_set.hist()
# plt.show()

# Data visualization (Multivariate  Plots)
# Scatter plot matrix
# scatter_matrix(data_set)
# plt.show()
# --------------------------------------------------------------------------------------------------------------------

# Split-out validation data set
array = data_set.values
x = array[:, 0:4]
y = array[:, 4]
validation_size = 0.20
seed = 7

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size,
                                                                                random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot check algorithms
models = list()
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = list()
names = list()

for name, model in models:
    k_fold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scoring)

    results.append(cv_results)
    names.append(name)

    msg = '{}: {} ({})'.format(name, cv_results.mean(), cv_results.std())
    # print(msg)


# Compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
# plt.show()

# Make predictions on validation data set
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_validation)

print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
