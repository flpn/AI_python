import warnings
from collections import Counter
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt


style.use('fivethirtyeight')


def k_nearest_neighbors(data_set, predict, k=3):
    if len(data_set) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []

    for group in data_set:
        for features in data_set[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [group[1] for group in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


data_set = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

result = k_nearest_neighbors(data_set, new_features)

[[plt.scatter(feature[0], feature[1], color=group) for feature in data_set[group]] for group in data_set]
plt.scatter(new_features[0], new_features[1], color=result)
plt.show()
