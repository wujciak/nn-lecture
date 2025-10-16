import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
n_redundant=0, n_repeated=0, random_state=None, class_sep=.7)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

clf = GaussianNB().fit(X, y)

n_samples = 300
nx = np.linspace(X[:, 0].min(), X[:, 0].max(), n_samples)
ny = np.linspace(X[:, 1].min(), X[:, 1].max(), n_samples)

xx, yy = np.meshgrid(nx, ny)
map = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
print(map, map.shape)

map_pred = clf.predict_proba(map)
distance = map_pred[:, 0] - .5

ax.scatter(map[:, 0], map[:, 1], c=distance, cmap="coolwarm")
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr_r")

plt.tight_layout()
plt.savefig('lecture1/foo.png')