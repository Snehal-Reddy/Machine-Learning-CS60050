import pandas as pd
import numpy as np
# np.random.seed(2)

def dist_(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

data = pd.read_csv('data4_19.csv', names=['sl','sw','pl','pw','class_'])

f1 = data['sl'].values
f2 = data['sw'].values
f3 = data['pl'].values
f4 = data['pw'].values
X = np.array(list(zip(f1, f2, f3, f4)))
clusters = np.zeros(len(X))


ind = np.random.choice(len(X), 3)
centroids = X[ind]
old_cen = X[ind]

from copy import deepcopy

for iter in range(10):
	for i in range(len(X)):
		dist = dist_(X[i],centroids)
		c = np.argmin(dist)
		clusters[i] = c

	old_cen = deepcopy(centroids)

	for i in range(3):
		points = [X[j] for j in range(len(X)) if clusters[j] == i]
		centroids[i] = np.mean(points, axis=0)

print("\nThe centroids are - ")
F = []
for i in range(3):
	print("Cluster " + str(i) + " -> " + str(centroids[i]) )
	F.append(set([j for j in range(len(X)) if clusters[j] == i]))
# A = set([j for j in range(len(X)) if clusters[j] == 0])
# B = set([j for j in range(len(X)) if clusters[j] == 1])
# C = set([j for j in range(len(X)) if clusters[j] == 2])
print("\n")
classes = data.class_.unique()
G = []
for i in classes:
	G.append(set(data[data['class_']==i].index.tolist()))

print("jaccard distances : ")
for i in range(3):
	max_ = 0
	for j in range(3):
		if ( (len(F[i]&G[j]))/(len(F[i])+len(G[j])-(len(F[i]&G[j]))) > max_ ):
			max_ = (len(F[i]&G[j]))/(len(F[i])+len(G[j])-(len(F[i]&G[j])))

	print("For cluster " + str(i) + " -> " + str(1-max_))