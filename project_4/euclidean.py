from math import sqrt
import numpy as np
import matplotlib.pyplot as plt 
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

"""
ornek euclidean distance bulma methodu

plot_1 = [2,5]
plot_2 = [5,9]

euclidean_distance = sqrt((plot_1[0]-plot_2[0])**2 + (plot_1[1]-plot_2[1])**2)
print("Euclidean distance:",euclidean_distance)
"""

# kullandigimiz stil, dataset ve ornek x, y noktasi
style.use("fivethirtyeight")
dataset = {"a" : [[1,2], [2,3], [3,4]], "b" : [[6,7], [7,8], [8,9]]}
sample_x_and_y = [4.5,5.5]

# en yakin nokta grubunun (k nearest neighbors) oldugu keyi donduruyoruz
def find_k_nearest_neighbors(dataset, sample, k):
    # veri kumesi, k den kucuk olmalı
    if(len(dataset) >= k):
        warnings.warn("Data lenght is not bigger than the k value!")

    # euclidean distance ve key leri diziye ekle
    distances = []
    for group in dataset:
        for x_and_y in dataset[group]:
            euclidean_distance = np.linalg.norm(np.array(sample)-np.array(x_and_y))
            distances.append([euclidean_distance, group])
            # distances.append([np.sqrt(np.sum(np.array(sample) - np.array(x_and_y))**2), group])
            # distances.append([sqrt((sample[0]-x_and_y[0])**2 + (sample[1]-x_and_y[1])**2), group])

    # uzakliklari sırala ve key leri diziye ekle
    votes = []
    for i in sorted(distances)[:k]:
        votes.append(i[1])

    # en cok tekrar eden keyi dondur
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = find_k_nearest_neighbors(dataset, sample_x_and_y, k=4)
print("The closest point in our dataset:", result)    

# dataset
for group in dataset:
    for x_and_y in dataset[group]:
        plt.scatter(x_and_y[0], x_and_y[1], s=75, color="b")

# ornek x, y noktasi
plt.scatter(sample_x_and_y[0], sample_x_and_y[1], s=100, color="r")
plt.show()

# veriyi oku, id column cikar, na verilerini -99999 ile doldur, float listesine cevir ve karistir
df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
list_datas = df.astype(float).values.tolist()
random.shuffle(list_datas)

# train_data 80% olmak uzere verileri olustur
test_size = 0.2
train_set = { 2 : [], 4 : []} 
test_set = { 2 : [], 4 : []}
train_data = list_datas[:-int(test_size*len(list_datas))]
test_data = list_datas[-int(test_size*len(list_datas)):]

# ilgili 2 ve 4 classlari icin dagitimi burada gerceklestir
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

# test_set icindeki verileri, train_set icindeki verilerle kiyasla ve dogruluk oranini yazdir
total = 0
correct = 0
for group in test_set:
    for data in test_set[group]:
        closest_group = find_k_nearest_neighbors(train_set, data, k=5)
        if closest_group == group:
            correct += 1
        total += 1

print("Accuracy in the breast-cancer-winconsin.data:", (correct/total))