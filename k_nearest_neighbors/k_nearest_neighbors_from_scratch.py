import random
import warnings
from collections import Counter
import numpy as np
import pandas as pd


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


data_frame = pd.read_csv('breast-cancer-wisconsin.txt')
data_frame.replace('?', -99999, inplace=True)
data_frame.drop('id', 1, inplace=True)

full_data = data_frame.astype(float).values.tolist()

random.shuffle(full_data)
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, 5)

        if group == vote:
            correct += 1

        total += 1

print(correct / total)
