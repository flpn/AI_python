from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# y = mx + b
# m -> best fit slope
# b -> y intercept

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (
        ((mean(xs) * mean(ys)) - mean(xs * ys)) /
        (mean(xs) ** 2 - mean(xs ** 2))
    )

    b = mean(ys) - m * mean(xs)

    return m, b


def squared_error(ys_original, ys_line):
    return sum((ys_line - ys_original) ** 2)


def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for _ in ys_original]
    squared_error_regression = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)

    return 1 - (squared_error_regression / squared_error_y_mean)


m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [m * x + b for x in xs]

predict_x = 8
predict_y = m * predict_x + b

# How accurate the best fit line is
coefficient_of_determination = coefficient_of_determination(ys, regression_line)
print(coefficient_of_determination)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
