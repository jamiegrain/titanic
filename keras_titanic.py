import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np 
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import Imputer, StandardScaler
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

model = Sequential()
model.add(Dense(units=500, activation=tf.nn.relu, input_dim=9))
model.add(Dense(units=200, activation=tf.nn.relu))
model.add(Dense(units=2, activation=tf.nn.softmax))

model.compile(loss='sparse_categorical_crossentropy',
	optimizer='sgd',
	metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=9)

loss, accuracy = model.evaluate(X_validation, y_validation)

df_test = pd.read_csv('test.csv', index_col = 0)

create_new_cols(df_test)
X_sub = preprocess(df_test)
y_sub = model.predict(X_sub)
y_sub = np.argmax(y_sub, axis = 1)

print(y_sub)

predictions = pd.DataFrame(y_sub, index = df_test.index)
predictions.rename(columns = {0: 'Survived'}, inplace = True)

print(predictions.head())


predictions.to_csv('titanic_submission.csv')
sub_check = pd.read_csv('titanic_submission.csv', index_col = 0)
print(sub_check.head())