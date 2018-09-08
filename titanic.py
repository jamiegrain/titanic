import numpy as np 
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV


df = pd.read_csv('train.csv', index_col = 0)

def fare_sqrt(fare):
	return fare**0.5

def create_new_cols(df):
	df['fare_sqrt'] = list(map(fare_sqrt, df['Fare']))

def separate_labels(df):
	y = df['Survived'].values
	df.drop('Survived', axis=1, inplace = True)
	return y, df

def preprocess(df):
	create_new_cols(df)
	df['Sex'].replace(['male', 'female'], [0, 1], inplace = True)
	df.drop(['Cabin', 'Name', 'Ticket', 'Fare'], axis=1, inplace=True)
	df = pd.get_dummies(df, prefix = ['Departed'], columns = ['Embarked'])

	pipeline = Pipeline([
		('imp', Imputer(strategy = 'most_frequent')),
		('scaler', StandardScaler())
		])

	X = pipeline.fit_transform(df)
	df = pd.DataFrame(X, columns = list(df))

	return X

print(df.corr()['Survived'])

y, df = separate_labels(df)
X = preprocess(df)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

def rbf_svc_fit(X, y):
	clf = SVC(kernel='rbf', gamma = 5, C = 1)
	clf.fit(X, y)
	return clf

def decision_tree_fit(X, y):
	clf = DecisionTreeClassifier(min_samples_leaf = 4)
	clf.fit(X, y)
	return clf

def gaussian_naive_bayes_fit(X, y):
	clf = GaussianNB()
	clf.fit(X, y)
	return clf

def mlp_fit(X, y):
	clf = MLPClassifier(hidden_layer_sizes = (1000, 300), random_state = 42)
	clf.fit(X, y)
	return clf

def fit_all_clfs(X, y):
	classifiers = [('svc', rbf_svc_fit(X, y)),
		('dtree', decision_tree_fit(X, y)),
		('gnb', gaussian_naive_bayes_fit(X, y)),
		('mlp', mlp_fit(X,y))]
	return classifiers

def ensemble_fit(X, y):
	voting_clf = VotingClassifier(estimators = fit_all_clfs(X,y), voting = 'hard')
	voting_clf.fit(X,y)
	return voting_clf

final_clf = ensemble_fit(X, y)
y_pred = final_clf.predict(X_validation)
print(f1_score(y_validation, y_pred))


df_test = pd.read_csv('test.csv', index_col = 0)

create_new_cols(df_test)
X_sub = preprocess(df_test)
y_sub =final_clf.predict(X_sub)

predictions = pd.DataFrame(y_sub, index = df_test.index)
predictions.rename(columns = {0: 'Survived'}, inplace = True)

print(predictions.head())


predictions.to_csv('titanic_submission.csv')
sub_check = pd.read_csv('titanic_submission.csv', index_col = 0)
print(sub_check.head())








