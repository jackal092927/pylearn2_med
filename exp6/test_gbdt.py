from __future__ import division
import cPickle
import gzip
# import matplotlib.pyplot as plt
from pylearn2.expr.preprocessing import global_contrast_normalize
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

__author__ = 'Jackal'
from sklearn.svm import l1_min_c, SVC, LinearSVC
from datasets.cin_feature2_composite import *
from sklearn import svm, linear_model, ensemble
import numpy as np
from sklearn.metrics import *
from utils.loader import *

params = {'n_estimators': 3000, 'learning_rate': 0.1, 'random_state': 13}
def basic_gbdt(trainset, testset):
    X_train, y_train = trainset
    X_test, y_test = testset

    # l = [('No shrinkage', 'orange',
    #       {'learning_rate': 1.0, 'subsample': 1.0}),
    #      ('learning_rate=0.1', 'turquoise',
    #       {'learning_rate': 0.1, 'subsample': 1.0}),
    #      ('subsample=0.5', 'blue',
    #       {'learning_rate': 1.0, 'subsample': 0.5}),
    #      ('learning_rate=0.1, subsample=0.5', 'gray',
    #       {'learning_rate': 0.1, 'subsample': 0.5}),
    #      ('learning_rate=0.1, max_features=2', 'magenta',
    #       {'learning_rate': 0.1, 'max_features': 2})]


    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    accur = clf.score(X_test, y_test)
    result = clf.predict(X_test)
    sensi, speci = my_scores(y_test, result)
    print "result:"
    print result
    print "accur:\t{} sensi:\t{} speci:\t{}".format(accur, sensi, speci)
    return accur, sensi, speci


def gbdt_cv(trainset, testset, scaler=None):
    trainx, trainy = trainset
    testx, testy = testset
    if scaler:
        trainx = scaler.fit_transform(trainx)
        trainy = scaler.fit_transform(trainy)
    lrs = [0.1]
    n_estimators = [3000]
    rand_stats = [1, 101, 1001]
    params = [{'learning_rate': lrs, 'n_estimators': n_estimators, 'random_state': rand_stats}]
    clf = GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid=params, cv=9)
    clf.fit(trainx, trainy)
    print clf.best_estimator_

    train_score = 0
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
        train_score = max(train_score, mean_score)
    res = clf.predict(testx)
    print "result:"
    print res
    test_score = clf.score(testx, testy)
    print "Train score:\t{}".format(train_score)
    print "Test score:\t{}".format(test_score)
    return train_score, test_score






def test_gbdt():
    key = '850+556'
    datapaths = datapath_helper(key)
    train_scos = []
    test_scos = []
    for i in range(1):
        datapath = datapaths[i]
        print '#' * 53
        print datapath
        trainset, testset = get_dataset(data_path=datapath)
        train_sco, test_sco = gbdt_cv(trainset, testset)
        train_scos.append(train_sco)
        test_scos.append(test_sco)
    print "train_scores:"
    print train_scos
    print "mean:\t{} std:\t{}".format(np.mean(train_scos), np.std(train_scos))
    print "test_scores:"
    print test_scos
    print "mean:\t{} std:\t{}".format(np.mean(test_scos), np.std(test_scos))


def main():
    key = '850+556'
    datapaths = datapath_helper(key)
    print params
    accurs = []
    sensis = []
    specis = []
    for i in range(10):
        data_path = datapaths[i]
        print '#' * 53
        print data_path
        trainset, testset = get_dataset(data_path=data_path)
        error, sensi, speci = basic_gbdt(trainset, testset)
        accurs.append(error)
        sensis.append(sensi)
        specis.append(speci)
    print accurs
    print "mean:\t{}\tstd:\t{}".format(np.mean(accurs), np.std(accurs))
    print "sensi:\t{}".format(np.mean(sensis))
    print "speci:\t{}".format(np.mean(specis))


if __name__ == '__main__':

    main()
    print "#####DONE#####"
