#Introduction
Contains ML model to detect fraud in transaction given data of first transaction of ecommerce site
		Files:

		'Fraud_data.csv': Dataset of first transaction of users on site
		'IpAddress_to_Country.csv': csv storing the mapping of ip addresses to countries
		'main.py': extracts features from data and train the classifier, also dump the classifier using joblib.This also segregates training    			and testing data into appropriate form and save them to csv files.
		'train_data.csv': data used to train the model 80% of original
		'train_target.csv': target values for training data (fraud or non-fraud class)
		'test_data.csv': data used for testing 20% of original dataset
		'test_target.csv': target values for test data
		'model': folder in which trained model is dumped
		'metrics': file storing the results of various models which have been tried i.e. their accuracy, precision, recall etc. 
			Out of all these Random forest performed best
		'test.py': File that can be used to load the model and test the performance on test data

#Satisfying requirements and testing
All the requirements have been specified in 'requirements.txt' file
Please install it before testing
   $ pip install -r requirements.txt

For testing only purpose (model testing)
run the file : python test.py
This will print the performance metrics in the terminal

#Reason for model selection and features used
Upon observing the data it was found that in many fraud cases the difference between signup_time and purchase_time is small.
Hence, this time diff has been used as a feature.
Also, location has been extraced from ip_address to location csv and used as a feature

user_id, device_id, signup_time, purchase_time have been dropped.

Dealing with Imbalanced data:
Since the data is imbalanced there are proportionally less no. of fraud cases as compared to non-fraud.
Various methods to deal with this were used:
1. Recall(how many of fraud cases get identified) and other measures were used to evaluate model.
2. Undersampling of majority class data was tried
3. Oversampling with SMOTE approach gave the best results

Please refer to 'metrics' file to see various results.

Model selection:
Since this is a classification task SVM and Decision tree based classifiers are most likely to give good result.
Both were tried. Since the data has many categorical fields like sex, location, browser, source Decision tree based classifiers will perform better and they did so as compared to svm classifier.
If we can tune in the parameters of svm model it can perform better.

Random forest ensemble over Decision tree was tried and it gave the best performance of all. The parameters might be needed to tune once we test on more expternal data. Parameter tuning has not been done as of now, model is giving good results and tuning might overfit.

Please refer to 'metrics' file for checking the performance of various models.
Best Performance Random forest:
Recall is,  0.939317458576
Precision is,  0.998638926696
Accuracy is,  0.969079127498
F1 score is,  0.968070268029
Kappa score is,  0.938150857468
Confusion matrix,
[[27411    35]
 [ 1659 25680]]
