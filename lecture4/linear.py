import itertools
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=10, n_features=3, n_informative=3, n_targets=1, random_state=None)
X = X[:, 0]
y = y/10

aa = list(range(0, 20, 1))
bb = -4

SR_line = []
#for a,b in itertools.product(aa, bb):
    #ax[0].plot(X, a*X + b, color='gray', alpha=0.2)
    # cośtamcośtam

# SR = np.sum((a*X+b - y)**2)

def linear_regression(X, y):
    n = len(y)
    mean_x , mean_y = np.mean(X), np.mean(y)
    
    SS_xy = np.sum(y*X) - n*mean_y*mean_x
    SS_xx = np.sum(X*X) - n*mean_x*mean_x
    
    a = SS_xy / SS_xx
    b = mean_y - a*mean_x

    return a, b


a, b = linear_regression(X, y)
print(a, b)
