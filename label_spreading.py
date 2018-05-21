import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.semi_supervised import label_propagation, LabelSpreading

train_labeled = pd.read_csv("train_labeledCSV.csv")
train_unlabeled = pd.read_csv("train_unlabeledCSV.csv")
test = pd.read_csv("testCSV.csv")

train_labeled = train_labeled.values
train_unlabeled = train_unlabeled.values
test = test.values

y_train_labeled = train_labeled[:9000,1]
x_train_labeled = train_labeled[:9000,2:]
x_train_unlabeled = train_unlabeled[:21000,1:]
minus_one = np.full((21000,1), -1, dtype=np.int)
y = np.concatenate((y_train_labeled.reshape(9000,1), minus_one),axis=0).ravel()
X = np.concatenate((x_train_labeled, x_train_unlabeled),axis=0)
print("started")
#ls = label_propagation.LabelSpreading().fit(X, y)
ls = LabelSpreading(kernel='knn', gamma=20, n_neighbors=7, alpha=0.0, max_iter=30, tol=0.001, n_jobs=1).fit(X, y)

sol = ls.predict(test[:,1:])

n = sol.shape[0]
y_solL = np.zeros(n)
y_solR = np.zeros(n)
for i in range(0,n):
    y_solL[i] = 30000+i
    y_solR[i] = sol[i]
y_sol = list(zip(y_solL,y_solR))

df = pd.DataFrame(y_sol)
df.to_csv("sol.csv",',',index=None,header=["Id","y"],float_format='%.0f')
