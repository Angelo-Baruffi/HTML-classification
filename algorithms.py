# -*- coding: utf-8 -*-
"""

@author: Angelo Baruffi e Andrei Donati
"""
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from time import time




# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [ 1e4],
              'gamma': [  0.02], }
clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred))
a= confusion_matrix(y_test, y_pred)





from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

clf = RandomForestClassifier(n_estimators=600, min_samples_split=5,criterion='entropy'  )
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
b = confusion_matrix(y_test, y_pred)

