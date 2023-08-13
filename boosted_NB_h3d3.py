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
	if verbose==1:
		visualize(Data, y_pred, y_pred_in, y_pred_out)


def visualize(Data, y_pred, y_pred_in, y_pred_out):
	date_list = Data[Data.columns[0]]
	x_tick_positions = [60 * i+5 for i in range(10)]
	date_ticks = [date_list[x] for x in x_tick_positions]
	plt.xticks(x_tick_positions, date_ticks, rotation=70)

	x_out = [len(y_pred_in) + i for i in range(len(y_pred_out))]

	ddd = DataFrame()
	ddd["Dates"] = date_list
	ddd["nber"] = y
	ddd["Recession Forecast"] = np.append(y_pred_in, y_pred_out)
	ddd.head(10)

	ddd["new_date"] = pd.to_datetime(ddd["Dates"], format="%Ym%m")
	ddd = ddd.sort_values(by="new_date")

	plt.plot(x_out, ddd["Recession Forecast"][len(y_pred_in):] , color = 'red', label="Out of Sample")
	plt.plot(ddd["Recession Forecast"][:len(y_pred_in)], color = 'green', label="In sample")

	plt.bar([i for i in range(len(y_pred))], ddd["nber"], width=1, color = 'gray')
	plt.title('Forecast for 3 months ahead horizon (h=3, d=3)', pad=30)

	plt.legend(loc="upper left")
	plt.show()


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
# Gaussian NB
print('GaussianNB')
model1 = GaussianNB()
model1.fit(X_train, y_train)
evaluate(model1, Data, X_train, X_test, y_test, verbose=0)

# Adaboost and NB
print('\nBoosted GaussianNB')
nb = GaussianNB()
model2 = AdaBoostClassifier(base_estimator=nb, n_estimators=500, algorithm="SAMME")
model2.fit(X_train, y_train)
evaluate(model2, Data, X_train, X_test, y_test, verbose=0)
