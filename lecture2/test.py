# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html

from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from onenn import OneNN
from sklearn.linear_model import LogisticRegression
import numpy as np

# # Remember to always specify random_state for reproducibility
# X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
#                             n_repeated=0, n_informative=2, random_state=1410)

# X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                         test_size=0.33, random_state=42, class_sep=0.3)

# clf = OneNN().fit(X_train, y_train)
# preds = clf.predict(X_test)
# score = accuracy_score(y_test, preds)
# print(score)


# Remember to always specify random_state for reproducibility
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                            n_repeated=0, n_informative=2, random_state=1410)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1410)

# CLFS x FOLDS
scores = np.zeros((2, 4))
for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    X_train, y_train = X[train_index], X[train_index]
    X_test, y_test = y[test_index], y[test_index]
    print(i)

    clf = OneNN().fit(X_train, y_train)
    preds = clf.predict(X_test)
    score = accuracy_score(y_test, preds)
    print(score)
    scores[0, i] = score

    clf = LogisticRegression().fit(X_train, y_train)
    preds = clf.predict(X_test)
    score = accuracy_score(y_test, preds)
    print(score)
    scores[1, i] = score

print(scores)
np.savetxt("exp_results.npy", scores)