import numpy as np

# scores = np.loadtxt("exp_results.npy")
# print(scores)

# SPECIFY DIMENSIONS OF ARRAY/TENSOR IT IS HELPFUL

# DATASETS x CLF x FOLDS x PREPROCESS x BUDGET
scores = np.random.randint(-9999, 9999, (20, 8, 10, 15, 30))
print(scores.shape)

# DATASETS x CLF x PREPROCESS x BUDGET
scores = np.mean(scores, axis=2)  # mean over folds
print(scores.shape)

# DATASETS x CLF x BUDGET
scores = scores[:, :, 0]
print(scores.shape)
