from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import operator
import skflow
import exp

sns.set_style("whitegrid")
sns.set_palette("bright")

###############################################
# PREPROCESS
###############################################

def preprocess(X, Y):
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X, Y = shuffle(X, Y, random_state=random.randint(0, 1000))
    return X, Y, scaler

def predict(clf, X_train, Y_train, X_test, Y_test):
    X_train, Y_train, scaler = preprocess(X_train, Y_train)
    clf.fit(X_train, Y_train)
    X_test = scaler.transform(X_test)
    Y_pred = clf.predict(X_test)
    score = metrics.r2_score(Y_test, Y_pred)
    print "R2-score: ", score
    return Y_pred

def visualize_preds(df, Y_test, Y_pred, test_years=None):
    if test_years:
        df = exp.filter_df_by_years(df, test_years)
    x = df["doy"]
    plt.scatter(x, Y_test, color='green', label="True value")
    plt.scatter(x, Y_pred, color='red', label="Predicted value")
    plt.legend(loc='best')
    plt.xlabel("doy")
    plt.show()

#########################################
# RANDOM FORESTS
#########################################

def random_forests():
    return RandomForestRegressor(n_estimators=200, max_features='sqrt', oob_score=True)

def random_forests_cross_val(X, Y, feature_names=None, k=10):
    print "Running Random Forests Cross Validation..."
    regr = random_forests()
    cv_scores = cross_val_score(regr, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean()
    print "CV Scores: ", ", ".join(map(str, cv_scores))
    regr.fit(X, Y)
    print "OOB score:", regr.oob_score_
    if feature_names:
        sorted_feature_importances = sorted(zip(feature_names, regr.feature_importances_), \
                                        key=operator.itemgetter(1), reverse=True)
        print "Feature Importances:"
        print '\n'.join(map(str, sorted_feature_importances))
    return regr

#########################################
# GRADIENT BOOSTED TREES
#########################################

def xgb_trees():
    return GradientBoostingRegressor(n_estimators=200, max_features='sqrt')

def xgb_trees_cross_val(X, Y, feature_names=None, k=10):
    print "Running Gradient Boosted Trees Cross Validation..."
    regr = xgb_trees()
    cv_scores = cross_val_score(regr, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean()
    print "CV Scores: ", ", ".join(map(str, cv_scores))
    regr = regr.fit(X, Y)
    if feature_names:
        sorted_feature_importances = sorted(zip(feature_names, regr.feature_importances_), \
                                        key=operator.itemgetter(1), reverse=True)
        print "Feature Importances:"
        print '\n'.join(map(str, sorted_feature_importances))
    return regr

#########################################
# SVM
#########################################

def svm():
    return SVR()

def svc_cross_val(X, Y, k=10):
    print "Running SVC Cross Validation..."
    regr = svm()
    cv_scores = cross_val_score(regr, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean()
    print "CV Scores: ", ", ".join(map(str, cv_scores))
    regr = regr.fit(X,Y)
    return regr

#########################################
# NEURAL NETWORK
#########################################

def dnn(nn_lr=0.1, nn_steps=5000, hidden_units=[30, 30]):
    def tanh_dnn(X, y):
        features = skflow.ops.dnn(X, hidden_units=hidden_units,
          activation=skflow.tf.tanh)
        return skflow.models.linear_regression(features, y)

    regressor = skflow.TensorFlowEstimator(model_fn=tanh_dnn, n_classes=0,
        steps=nn_steps, learning_rate=nn_lr, batch_size=100)
    return regressor

def dnn_cross_val(X, Y, k=10):
    print "Running Neural Network Cross Validation..."
    clf = dnn()
    cv_scores = []
    for train_indices, test_indices in KFold(X.shape[0], n_folds=k, shuffle=True, random_state=random.randint(0, 1000)):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        clf.fit(X_train, Y_train)
        score = metrics.r2_score(Y_test, clf.predict(X_test))
        cv_scores.append(score)
    print "{0}-fold CV Acc Mean: ".format(k), np.mean(cv_scores)
    print "CV Scores: ", ", ".join(map(str, cv_scores))
    return clf
