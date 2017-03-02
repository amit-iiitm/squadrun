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

#load the fraud dataset from csv to pandas dataframe
df_tran= DataFrame.from_csv("Fraud_Data.csv", index_col=False)
#load the ip_to_country mapping csv to a data frame
df_ip= DataFrame.from_csv("IpAddress_to_Country.csv", index_col=False)

#check the shape of dataframe
print df_tran.shape
print df_ip.shape[0]

#function to convert datetime format to timestamp
def totimestamp(dt, axis, epoch=datetime(1970,1,1)):
    #convert the datetime string to a datetime object
    dt=datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    #get the time difference between current and start refrence point
    td = dt - epoch
    #convert the time differnce to timestamp
    #** opearator means exponentiation
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6 


#function to get the corresponding country of an IP
def getcountry(dt,axis):
    #bisect can be used as the values of lower_bound_ip_add are stored in increasing sotrted order
    #bisect performs a binary search to get the index where this element can be put
    i=bisect.bisect(df_ip['lower_bound_ip_address'],dt)
    #if we reach the end of list for a certain value its country can be determined by value in last row
    if i>=df_ip.shape[0]:
        i=i-1
    print df_ip.loc[[i]]['country']
    #return the country that ip corresponds to
    return df_ip.loc[[i]]['country'].values[0]


def train():
    #Modify the dataset and extract features

    #first convert the datetime into timestamp using a function and apply it to whole series
    df_tran['signup_time']=df_tran['signup_time'].apply(totimestamp,axis=1)
    df_tran['purchase_time']=df_tran['purchase_time'].apply(totimestamp,axis=1)
    #upon observation it is seen that fraud cases have a small difference of time between first purchase and signup
    #use the difference of signup time and purchase time as a feature
    df_tran['time_diff']=df_tran['purchase_time'] - df_tran['signup_time']
    
    #extract the corresponding country of the ip address using other csv
    df_tran['location']=df_tran['ip_address'].apply(getcountry,axis=1)

    print "successful"
    #get the target class into a separate frame
    df_target=df_tran['class']
    #drop the signup,purchase time and user,device ids
    #as decision tree and its variants are used it will be good to drop ID fields
    df_data=df_tran.drop(['user_id','device_id','signup_time','purchase_time','ip_address','class'],axis=1)
    print df_target.shape
    print df_data.shape
    #df_data stores the data and df_target stores the target class

    #transform the locations onto numeric values
    #initialize a LabelEncoder obeject and use it to map categorical string values to numeric classes
    le=preprocessing.LabelEncoder()
    df_data['location']=le.fit_transform(df_data['location'])
    df_data['source']=le.fit_transform(df_data['source'])
    df_data['browser']=le.fit_transform(df_data['browser'])
    df_data['sex']=le.fit_transform(df_data['sex'])

    #Please note that the data is imbalanced and the proportion of fraud cases is very less than non-fraud
    #there are ideas to counter imbalance: under sampling of majority class or oversampling of minority

    #SMOTE approach tries to oversample minority class in a balanced way
    print "using SMOTE oversampling of minority class"
    #Initialize an smote object from imbalance learn module of python
    sm = SMOTE(kind='regular')
    #apply smote oversampling
    df_data, df_target=sm.fit_sample(df_data,df_target)

    #split into training and testing data(80:20 split)
    df_data_train, df_data_test, df_target_train, df_target_test=train_test_split(df_data,df_target,test_size=0.20,random_state=0)

    #choose a classifier model
    clf = RandomForestClassifier(n_estimators=200)
    #clf=tree.DecisionTreeClassifier()
    #clf=svm.SVC(class_weight='balanced')
    print "start training"
    clf.fit(df_data_train,df_target_train)
    print "training done"
    
    #dump the model for testing purposes
    joblib.dump(clf,'model/random_clf.pkl')
    print "model dumped"
    #use the model to predict upon the test data(20% of original)
    pred=clf.predict(df_data_test)

    #printing the evaluation metrices
    #accuracy can't be used as measure of model hence metrics like recall, precision, kappa are also measured
    print "Recall is, ", recall_score(df_target_test,pred)
    print "Precision is, ", precision_score(df_target_test,pred)
    print "Accuracy is, ", accuracy_score(df_target_test,pred)
    print "F1 score is, ", f1_score(df_target_test,pred)
    print "Kappa score is, ", cohen_kappa_score(df_target_test,pred)
    print "Confusion matrix, "
    print confusion_matrix(df_target_test,pred)

    #save the train and test data to csv files
    #using numpy.savetxt
    #seven features ['purchase_value','source','browser','sex','age','time_diff','location']
    numpy.savetxt('train_data.csv',df_data_train,delimiter=',')
    numpy.savetxt('train_target.csv',df_target_train,delimiter=',')
    numpy.savetxt('test_data.csv',df_data_test,delimiter=',')
    numpy.savetxt('test_target.csv',df_target_test,delimiter=',')
    print "save successful"

if __name__=='__main__':
    train()
