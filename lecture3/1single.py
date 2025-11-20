import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score as bac
from sklearn.base import clone

# keel a good site with datasets, do at least 20 of them to test

data = np.getfromtxt('datasets/australian.csv', delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)

clfs = {
    'GNB': GaussianNB(), 
    'LR': LogisticRegression(),
    'CART': DecisionTreeClassifier(random_state=1410),
    'kNN': KNeighborsClassifier()
}

# CLFS, FOLDS
scores = np.zeros((len(clfs), 4))  # 2 splits * 2 repeats = 4
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=1410)
for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    
    for clf_id, clf_name in enumerate(clfs.keys()):
        clf = clone(clfs[clf_name]).fit(X_train, y_train)
        preds = clf.predict(X_test)
        score = bac(y_test, preds)
        scores[clf_id, i] = score

np.save("scores1", scores)
