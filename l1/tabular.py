# Basic Taxononomy of ML are: supervised, unsupervised, reinforcement. 
# We will be talking about supervised learning here mostly.

# Supervised learning is further divided into classification and regression.
# Classification is when the output variable is a category, such as "red" or "blue".

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# y - labels (0, 1, 2)
# X - features (sepal length, sepal width, petal length, petal width)
# Iris is classic dtaset for flowers: https://scikit-learn.org/1.4/auto_examples/datasets/plot_iris_dataset.html
X, y = load_iris(return_X_y=True)
# print(X.shape) # printed (150, 4) so 150 samples = rows, 4 features = columns
columns = ['sepal length', 'sepal width', 'petal length', 'petal width']

# if we want to take all rows and 2 columns
#X = X[:, :2]

n_features = X.shape[1]

# fig, ax = plt.subplots(n_features, n_features, figsize=(10, 10))

# for i in range (n_features):
#     for j in range(n_features):
#         ax[i, j].scatter(X[:, i], X[:, j], c=y)
#         ax[i, j].grid(ls=":", c=(.7, .7, .7))
#         ax[i, j].set_xlabel(columns[i])
#         ax[i, j].set_ylabel(columns[j])


# if we have many dimensions, we can do feature extraction or feature selection
# feature extraction is when we create new features from the existing ones, some information is lot
# one of the most popular feature extraction techniques is PCA (principal component analysis)
# components are our new feature (reducted ones)
# pca = PCA(n_components=2).fit(X, y)
# print(pca.explained_variance_ratio_) # how much information we kept
# X = pca.transform(X)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(X[:, 0], X[:, 1], c=y)
# print(X)

ax.grid(ls=":", c=(.7, .7, .7))
plt.tight_layout()
plt.savefig('foo.png')