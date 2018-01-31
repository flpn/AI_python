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
        latest_aptimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_aptimum, latest_aptimum])
            optimized = False  # we can do this because convex

            while not optimized:
                pass

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
