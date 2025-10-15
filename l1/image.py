import numpy as np
import os
import matplotlib.pyplot as plt
from cv2 import resize


classes = ["gatto", "pecora"] # not present here

X = []
y = []
for class_id, cls in enumerate(classes):
    files = os.listdir("image/%s" % cls)
    print(files)
    for img_id, file in enumerate(files[:100]):
        img = plt.imread("images/%s/%s" % (cls, file))
        img = resize(img, (224, 224)).flatten()
        X.append(class_id)

X = np.array(X)
