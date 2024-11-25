import math

import numpy as np
#
A = [20,8,-4,4]
B = [5,-2,-3,4]
# #
# A1 = [20,+8,-6,+6]
# B1 = [5,-2,-3,+4]

data = np.array([A,B])

covMatrix = np.cov(data,bias=True)
print (covMatrix)
#
import scipy.stats as stats


data = np.array([stats.zscore(A),stats.zscore(B)])
print(data)

import pandas as pd


df = pd.DataFrame([A,B])

corrMatrix = df.corr()
print (np.corrcoef(A, B))

A = np.mat("75 -21;-21 12.5")
print ("A\n", A)
print ("Eigenvalues", np.linalg.eigvals(A))
eigenvalues, eigenvectors = np.linalg.eig(A)
print ("eigenvectors", eigenvectors)



# Importing Libraries
# import numpy as np
# import matplotlib.pyplot as plt
#
#
def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost
#

x=3
y=3
w=0
n=0.1
for i in range(3):
    # w_=2*(math.exp(w*x)-y)*x*math.exp(w*x)
    w_=2*x*(w*x-y)
    w=w-n*w_
    print("w"+str(i+1)+":"+str(w))