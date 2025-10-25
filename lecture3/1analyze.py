import numpy as np
import scipy.stats as stats
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate

def cv52cft(a, b):
    d = a.reshape(2, 5) - b.reshape(2, 1)
    f = np.sum(np.power(d, 2)) / (2 * np.sum(np.var(d, axis=0, ddof=0)))
    p = 1 - stats.f.cdf(f, 10, 5)
    return f, p

clfs = {
    'GNB': GaussianNB(), 
    'LR': LogisticRegression(),
    'CART': DecisionTreeClassifier(random_state=1410),
    'kNN': KNeighborsClassifier()
}

# CLFS, FOLDS
scores = np.load("scores1.npy")
# print(scores, scores.shape)

alfa = .05
f_statistic = np.zeros(len(clfs), len(clfs))
p_value = np.zeros(len(clfs), len(clfs))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        f_statistic[i, j], p_value[i, j] = cv52cft(scores[i], scores[j])

print(tabulate(f_statistic, tablefmt="simple", floatfmt=".3f"))
# print(tabulate(p_value, tablefmt="simple", floatfmt=".3f")) #  tablefmt="latex_booktabs" very useful for papers in latex

advantage = np.zeros(len(clfs), len(clfs))
advantage[f_statistic > 0] = 1
print(advantage) # which is better

significance = np.zeros(len(clfs), len(clfs))
significance[p_value <= alfa] = 1
print(significance) # which is statistically significant


stat_better = significance * advantage
print(np.mean(scores, axis=1))
print(stat_better) # which is statistically significantly better
