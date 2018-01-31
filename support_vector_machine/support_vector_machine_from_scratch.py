import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt


style.use('ggplot')


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        self.fig, self.ax = self.is_visual()

    def fit(self, data):
        self.data = data

        # {||w||: [w, b]}
        opt_dict = {}
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        all_data = []

        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [
            self.max_feature_value * 0.1,
            self.max_feature_value * 0.01,
            self.max_feature_value * 0.001,  # point of expense
        ]

        # extreme expensive
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False  # we can do this because convex

            while not optimized:
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transformations in transforms:
                        w_transformation = w * transformations
                        found_option = True

                        # weakest link in the SVM fundamentally; SMO attempts to fix this a bit; yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i

                                if not yi * (np.dot(w_transformation, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_transformation)] = [w_transformation, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w -= step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification

    def is_visual(self):
        if self.visualization:
            return plt.figure(), self.fig.add_subplot(1, 1, 1)

        return None, None


data_dict = {
    -1: np.array([[1, 7], [2, 8], [3, 8], ]),
    1: np.array([[5, 1], [6, -1], [7, 3], ])}
