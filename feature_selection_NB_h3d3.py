import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def evaluate(model, Data, X_train, X_test, y_test, verbose=0):
	y_pred_out = model.predict(X_test)
	y_pred_in = model.predict(X_train)
	y_pred = np.append(y_pred_in, y_pred_out)
	cm = metrics.confusion_matrix(y_test, y_pred_out)
	print(cm)
	test_acc = metrics.accuracy_score(y_test, y_pred_out)
	print("Accuracy:", test_acc)
	f1_score = metrics.f1_score(y_test, y_pred_out)
	print('F1:', f1_score)
	precision_score = metrics.precision_score(y_test, y_pred_out)
	print('Precision:', precision_score)


def fix_data_with_nan(Data):
	def nan_helper(y):
		return np.isnan(y), lambda z: z.nonzero()[0]
	new_data = DataFrame()
	columns_to_copy = ["modate"]
	for column in columns_to_copy:
		new_data[column] = Data[column]
	for column in Data.columns:
		if column in columns_to_copy: continue
		y = np.array(Data[column])
		nans, x = nan_helper(y)
		y[nans] = np.interp(x(nans), x(~nans), y[~nans])
		new_data[column] = y
	return new_data


print('Loading data...')
# Data
Data= pd.read_csv('data_h3d3.csv',sep='\t')
Data = fix_data_with_nan(Data)
# Training and testing set size
train_size=int(0.75*Data.shape[0])
test_size=int(0.25*Data.shape[0])
print("Training set size : "+ str(train_size))
print("Testing set size : "+str(test_size))
# Getting features from dataset
Data= Data.sample(frac=1)
X = Data.iloc[:,2:401].values
y = Data.iloc[:,1].values
X = X.astype(float)
# Feature scaling
from FeatureScaling import FeatureScaling
fs = FeatureScaling(X,y)
X = fs.fit_transform_X()
# training set split
X_train=X[0:train_size,:]
y_train=y[0:train_size]
# testing set split
X_test=X[train_size:,:]
y_test=y[train_size:]


print('Training...')
# Adaboost and FeatureSelectionNB
from sklearn.base import BaseEstimator
class FeatureSelectionNB(BaseEstimator):
	def __init__(self, score_type='acc'):
		self.best_model = None
		self.selected_feature = -1
		self.score_type = score_type # 'f1', 'acc'

	def fit(self, x, y, sample_weight=np.array([False])):
		best_score = float("-inf")
		for i in range(x.shape[1]):
			nb = GaussianNB()
			selected_data = np.expand_dims(x[:,i], axis=1)
			if sample_weight.any():
				nb.fit(selected_data, y, sample_weight=sample_weight)
			else:
				nb.fit(selected_data, y)
			y_predict = nb.predict(selected_data)
			if self.score_type == 'f1':
				score = metrics.f1_score(y, y_predict)
			else: # 'acc'
				score = metrics.accuracy_score(y, y_predict)
			if score > best_score:
				best_score = score
				self.best_model = nb
				self.selected_feature = i
		self.classes_ = self.best_model.classes_
		return self

	def predict(self, x):
		selected_data = np.expand_dims(x[:, self.selected_feature], axis=1)
		return self.best_model.predict(selected_data)

print('FeatureSelectionNB')
fnb = FeatureSelectionNB()
fnb.fit(X_train, y_train)
evaluate(fnb, Data, X_train, X_test, y_test, verbose=0)

print('\nBoosted FeatureSelectionNB')
fnb = FeatureSelectionNB()
boosted_fnb = AdaBoostClassifier(base_estimator=fnb, n_estimators=10, algorithm="SAMME")
boosted_fnb.fit(X_train, y_train)
evaluate(boosted_fnb, Data, X_train, X_test, y_test, verbose=0)

print('\nGaussianNB')
model1 = GaussianNB()
model1.fit(X_train, y_train)
evaluate(model1, Data, X_train, X_test, y_test, verbose=0)

print('\nBoosted GaussianNB')
nb = GaussianNB()
model2 = AdaBoostClassifier(base_estimator=nb, n_estimators=500, algorithm="SAMME")
model2.fit(X_train, y_train)
evaluate(model2, Data, X_train, X_test, y_test, verbose=0)



