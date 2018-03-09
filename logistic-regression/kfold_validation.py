# kfold_validation.py
# logistic-regression with ROC Curve
# Shah Zafrani

from sklearn.model_selection import KFold
import numpy as np

# hyper-parameters
numFolds = 10
# learning_rate = 1e-6

# setting up data
mnist_data = np.genfromtxt('MNIST_CV.csv', delimiter=',', dtype=int, skip_header=1)
kf = KFold(n_splits=numFolds)
kf.get_n_splits(mnist_data)
numiter = 0
for train_index, test_index in kf.split(mnist_data):
    # print to make sure folding is working correctly
    # print("TRAIN:", train_index, "TEST:", test_index)
    # This will happen 10 times. Do your regression and save TPR and FPR here
