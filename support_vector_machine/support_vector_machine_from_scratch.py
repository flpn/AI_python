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
        pass

    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification

    def is_visual(self):
        if self.visualization:
            return plt.figure(), self.fig.add_subplot(1, 1, 1)

        return None, None


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),

             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}
