#!/usr/bin/python3

"""
@author: Maria Efthymiou
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
import time

def scores(ytrue, ypred):
    acc = metrics.accuracy_score(ytrue, ypred)
    pre = metrics.precision_score(ytrue, ypred)
    rec = metrics.recall_score(ytrue, ypred)
    return acc, pre, rec

# **************************** Dataset ****************************************
data = pd.read_csv("breastCancer.csv")
print("**********************************************************************")
print(data.info())
print("**********************************************************************")
print(data.describe())
print("**********************************************************************")
print(data.shape)
print("**********************************************************************")

# ************************* Preprocessing *************************************
data["diagnosis"].replace(["B", "M"], [0, 1], inplace=True)
X = data.drop(labels=["id", "diagnosis"], axis=1, inplace=False)
y = data['diagnosis']
print(X.describe())
print("**********************************************************************")
sclr = StandardScaler()
X = sclr.fit_transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y)

# ************************* SVM Classifier ************************************
svc = SVC()
s = GridSearchCV(svc, [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]).fit(X, y)
best_svc = s.best_estimator_
best_svc.fit(Xtrain, ytrain)
start_time = time.time_ns()
svc_res = best_svc.predict(Xtest)
svc_time = time.time_ns() - start_time
svc_acc, svc_pre, svc_rec = scores(ytest, svc_res)
#SVC defaul kernel is rbf
print("Best SVC model:", best_svc, "\nTime: ", svc_time, "nanoseconds")
print("Accuracy: %f, Precision: %f, Recall: %f" % (svc_acc, svc_pre, svc_rec))
print("**********************************************************************")

# ************************* Decision Tree *************************************
dtree = DecisionTreeClassifier()
s = GridSearchCV(dtree, [{"criterion" : ["gini", "entropy"]}]).fit(X, y)
best_tree = s.best_estimator_
best_tree.fit(Xtrain, ytrain)
start_time = time.time_ns()
tree_res = best_tree.predict(Xtest)
ftime = time.time_ns()
tree_time = ftime - start_time
tree_acc, tree_pre, tree_rec = scores(ytest, tree_res)
print("Best Decision Tree model:", best_tree, "\nTime: ", tree_time, "nanoseconds")
print("Accuracy: %f, Precision: %f, Recall: %f" % (tree_acc, tree_pre, tree_rec))
print("**********************************************************************")

# ************************* Random Forest *************************************
forest = RandomForestClassifier()
s = GridSearchCV(forest, [{"criterion" : ["gini", "entropy"]}]).fit(X, y)
best_forest = s.best_estimator_
best_forest.fit(Xtrain, ytrain)
start_time = time.time_ns()
forest_res = best_forest.predict(Xtest)
forest_time = time.time_ns() - start_time
forest_acc, forest_pre, forest_rec = scores(ytest, forest_res)
print("Best Random Forest model:", best_forest, "\nTime: ", forest_time, "nanoseconds")
print("Accuracy: %f, Precision: %f, Recall: %f" % (forest_acc, forest_pre, forest_rec))
print("**********************************************************************")

# ************************* KNN Classifier ************************************
knn = KNeighborsClassifier()
s = GridSearchCV(knn, [{"n_neighbors" : range(1, 11)}]).fit(X, y)
best_knn = s.best_estimator_
best_knn.fit(Xtrain, ytrain)
start_time = time.time_ns()
knn_res = best_knn.predict(Xtest)
knn_time = time.time_ns() - start_time
knn_acc, knn_pre, knn_rec = scores(ytest, knn_res)
print("Best K-Neighbors model:", best_knn, "\nTime: ", knn_time, "nanoseconds")
print("Accuracy: %f, Precision: %f, Recall: %f" % (knn_acc, knn_pre, knn_rec))
print("**********************************************************************")

# ************************* MLP Classifier ************************************
mlp = MLPClassifier(max_iter=800)
s = GridSearchCV(mlp, [{"activation" : ["identity", "logistic", "tanh", "relu"],
                        "solver" : ["lbfgs", "sgd", "adam"]}]).fit(X, y)
best_mlp = s.best_estimator_
best_mlp.fit(Xtrain, ytrain)
start_time = time.time_ns()
mlp_res = best_mlp.predict(Xtest)
mlp_time = time.time_ns() - start_time
mlp_acc, mlp_pre, mlp_rec = scores(ytest, mlp_res)
print("Best Multi-layer Perceptron model:", best_mlp, "\nTime: ", mlp_time, "nanoseconds")
print("Accuracy: %f, Precision: %f, Recall: %f" % (mlp_acc, mlp_pre, mlp_rec))
print("**********************************************************************")
