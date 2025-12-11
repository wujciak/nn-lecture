import numpy as np
import os
import matplotlib.pyplot as plt
from cv2 import resize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


classes = ["gatto", "pecora"] # not present here

X = []
y = []
for class_id, cls in enumerate(classes):
    files = os.listdir("../l1/image/%s" % cls)
    print(files)
    for img_id, file in enumerate(files[:100]):
        img = plt.imread("../l1/images/%s/%s" % (cls, file))
        img = resize(img, (224, 224)).flatten()
        X.append(class_id)

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pca = PCA(n_componentss=.7)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
clf = LogisticRegression()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
score = balanced_accuracy_score(y_test, preds)
print("Score:", score)
