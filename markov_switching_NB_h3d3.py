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
    test_acc = metrics.accuracy_score(y_test, y_pred_out)
    f1_score = metrics.f1_score(y_test, y_pred_out)
    precision_score = metrics.precision_score(y_test, y_pred_out)
    prob_pred = model.predict_proba(X_test)
    prob_pred = prob_pred[:, 1] / np.sum(prob_pred, axis=1)
    mae = metrics.mean_absolute_error(y_test, prob_pred)

    if verbose > 0:
        print('Confusion matrix:')
        print(cm)
        print('Accuracy:', test_acc)
        print('F1:', f1_score)
        print('Precision:', precision_score)
        print('MAE:', mae)

    return mae, test_acc


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


def data_split(Data):
    # Training and testing set size
    train_size = int(0.75 * Data.shape[0])
    test_size = int(0.25 * Data.shape[0])
    print("Training set size : " + str(train_size))
    print("Testing set size : " + str(test_size))
    # Getting features from dataset
    Data = Data.sample(frac=1)
    X = Data.iloc[:, 2:401].values
    y = Data.iloc[:, 1].values
    X = X.astype(float)
    # Feature scaling TODO
    from FeatureScaling import FeatureScaling
    fs = FeatureScaling(X, y)
    X = fs.fit_transform_X()
    # training set split
    X_train = X[0:train_size, :]
    y_train = y[0:train_size]
    # testing set split
    X_test = X[train_size:, :]
    y_test = y[train_size:]
    return X_train, y_train, X_test, y_test


def get_markov_switching_y_models(Data):
    '''
        Get prior model and transition model for recession in Data. With prediction horizon = 3
    '''
    y = Data.iloc[:, 1].values
    N = y.shape[0]
    y_prev = y[: N-1]
    y_prev = np.insert(y_prev, 0, 0)

    prior, transition = [0, 0], [[0, 0], [0, 0]]
    prior[1] = np.count_nonzero(y) / N
    prior[0] = 1.0 - prior[1]
    transition[0][1] = np.count_nonzero(y[y_prev == 0]) / y[y_prev == 0].shape[0] # transition model from y=0 to y=1
    transition[0][0] = 1.0 - transition[0][1]
    transition[1][1] = np.count_nonzero(y[y_prev == 1]) / y[y_prev == 1].shape[0]
    transition[1][0] = 1.0 - transition[1][1]
    return prior, transition

class MarkovSwitchingNB(GaussianNB):
    def set_markove_switching_model(self, priors, transitions, nber_4_index=398, nber_5_index=399, nber_6_index=400):
        self.priors = priors
        self.transitions = transitions
        self.nber_4_index = nber_4_index
        self.nber_5_index = nber_5_index
        self.nber_6_index = nber_6_index

    def _joint_log_likelihood(self, X):
        N = X.shape[0]
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            # Get Markov Switching models
            nber_4 = X[:, self.nber_4_index]
            nber_5 = X[:, self.nber_5_index]
            nber_6 = X[:, self.nber_6_index]

            prior_nber_6 = np.ones(N)
            prior_nber_6[nber_6 < 0] = self.priors[0]
            prior_nber_6[nber_6 > 0] = self.priors[1]

            transition_nber_6_to_nber_5 = np.ones(N)
            transition_nber_6_to_nber_5[np.logical_and(nber_6 < 0, nber_5 < 0)] = self.transitions[0][0]
            transition_nber_6_to_nber_5[np.logical_and(nber_6 > 0, nber_5 < 0)] = self.transitions[1][0]
            transition_nber_6_to_nber_5[np.logical_and(nber_6 < 0, nber_5 > 0)] = self.transitions[0][1]
            transition_nber_6_to_nber_5[np.logical_and(nber_6 > 0, nber_5 > 0)] = self.transitions[1][1]

            transition_nber_5_to_nber_4 = np.ones(N)
            transition_nber_5_to_nber_4[np.logical_and(nber_5 < 0, nber_4 < 0)] = self.transitions[0][0]
            transition_nber_5_to_nber_4[np.logical_and(nber_5 > 0, nber_4 < 0)] = self.transitions[1][0]
            transition_nber_5_to_nber_4[np.logical_and(nber_5 < 0, nber_4 > 0)] = self.transitions[0][1]
            transition_nber_5_to_nber_4[np.logical_and(nber_5 > 0, nber_4 > 0)] = self.transitions[1][1]

            transition_nber_4_to_nber_1 = np.ones(N)
            transition_nber_4_to_nber_1[nber_4 < 0] = self.transitions[0][i]
            transition_nber_4_to_nber_1[nber_4 > 0] = self.transitions[1][i]

            prior_nber_4 = np.ones(N)
            prior_nber_4[nber_4 < 0] = self.priors[0]
            prior_nber_4[nber_4 > 0] = self.priors[1]

            jointi = prior_nber_6 * transition_nber_6_to_nber_5 * transition_nber_5_to_nber_4 * transition_nber_4_to_nber_1
            # jointi = prior_nber_4 * transition_nber_4_to_nber_1
            jointi = np.log(jointi)
            #
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

def get_markov_switching_info(Data):
    # print(Data.head(6))
    # print('Nber4',Data.columns.get_loc('nber_4'))
    # print('Nber5',Data.columns.get_loc('nber_5'))
    # print('Nber6',Data.columns.get_loc('nber_6'))
    nber_4_index = Data.columns.get_loc('nber_4')
    nber_5_index = Data.columns.get_loc('nber_4')
    nber_6_index = Data.columns.get_loc('nber_4')
    prior_model, transition_model = get_markov_switching_y_models(Data)
    return prior_model, transition_model, nber_4_index, nber_5_index, nber_6_index


print('Loading data...')
# Data
Data = pd.read_csv('data_h3d3.csv', sep='\t')
priors, transitions, nber_4_index, nber_5_index, nber_6_index= get_markov_switching_info(Data)
Data = fix_data_with_nan(Data)
X_train, y_train, X_test, y_test = data_split(Data)
nber_4_index, nber_5_index, nber_6_index = nber_4_index, nber_5_index, nber_6_index

avg_mae_GNB = 0
avg_mae_MSNB = 0
avg_acc_GNB = 0
avg_acc_MSNB = 0
n_loop = 1
print('Training...')
for i in range(n_loop):
    # Gaussian NB
    # print('GaussianNB:')
    model1 = GaussianNB()
    model1.fit(X_train, y_train)
    mae, acc = evaluate(model1, Data, X_train, X_test, y_test, verbose=0)
    avg_mae_GNB += mae
    avg_acc_GNB += acc

    # Markov Switching NB
    # print('\nMarkovSwitchingNB:')
    model1 = MarkovSwitchingNB()
    model1.set_markove_switching_model(priors, transitions, nber_4_index, nber_5_index, nber_6_index)
    model1.fit(X_train, y_train)
    mae, acc = evaluate(model1, Data, X_train, X_test, y_test, verbose=0)
    avg_mae_MSNB += mae
    avg_acc_MSNB += acc

print('GNB VS MSNB')
print('MAE:', avg_mae_GNB/n_loop, avg_mae_MSNB/n_loop)
print('ACC:', avg_acc_GNB/n_loop, avg_acc_MSNB/n_loop)
