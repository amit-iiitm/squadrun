from __future__ import division
from datetime import datetime, timedelta
from pandas import DataFrame, merge, Series
from sklearn.ensemble import RandomForestClassifier
import numpy
import csv
import bisect
import sklearn
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
from sklearn import tree
from sklearn import svm, datasets, cross_validation
from sklearn import preprocessing

#imblearn module provides several ways to deal with the imbalance in data
from imblearn.over_sampling import SMOTE

#load the testdata
df_test_data=DataFrame.from_csv("test_data.csv", index_col=False)
df_test_target=DataFrame.from_csv("test_target.csv", index_col=False)

#print the shape of dataframes
print df_test_data.shape
print df_test_target.shape[0]


def test():
    #load the classifier
    clf=joblib.load('model/random_clf.pkl')
    #predict on the test data
    pred=clf.predict(df_test_data)

    #printing the evaluation metrices
    #accuracy can't be used as measure of model hence metrics like recall, precision, kappa are also measured
    print "Recall is, ", recall_score(df_test_target,pred)
    print "Precision is, ", precision_score(df_test_target,pred)
    print "Accuracy is, ", accuracy_score(df_test_target,pred)
    print "F1 score is, ", f1_score(df_test_target,pred)
    print "Kappa score is, ", cohen_kappa_score(df_test_target,pred)
    print "Confusion matrix, "
    print confusion_matrix(df_test_target,pred)

if __name__=='__main__':
    test()
