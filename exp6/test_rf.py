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
from sklearn import cross_validation


RANDOM_SEED = 1

def basic_rf(trainset, testset, n_estimators=1000):
    # cs = l1_min_c(trainX, trainy, loss='log') * np.logspace(0, 3)
    # print cs
    X_train, y_train = trainset
    X_test, y_test = testset
    errors = []
    sensis = []
    specis = []
    original_params = [{'n_estimators': n_estimators, 'max_depth': None,
                        'random_state': 2, 'min_samples_split': 5}]

    # plt.figure()

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

    l = [('No shrinkage', 'orange',
          {'learning_rate': 1.0, 'subsample': 1.0}),
         ('learning_rate=0.1', 'turquoise',
          {'learning_rate': 0.1, 'subsample': 1.0}),
         ('subsample=0.5', 'blue',
          {'learning_rate': 1.0, 'subsample': 0.5}),
         ('learning_rate=0.1, subsample=0.5', 'gray',
          {'learning_rate': 0.1, 'subsample': 0.5}),
         ('learning_rate=0.1, max_features=2', 'magenta',
          {'learning_rate': 0.1, 'max_features': 2})]

    l = [('No shrinkage', 'orange')]

    for params in original_params:
        # params = dict(original_params)
        # params.update(setting)
        # print params
        clf = ensemble.RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        result = clf.predict(X_test)
        error = np.count_nonzero(result - y_test) / X_test.shape[0]
        sensi, speci = my_scores(y_test, result)
        print error, sensi, speci
        return (error, sensi, speci)
        # compute test set deviance
        # test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

        # for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        #     # clf.loss_ assumes that y_test[i] in {0, 1}
        #     test_deviance[i] = clf.loss_(y_test, y_pred)


    #     print 'To plot'
    #     plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
    #              '-', color=color, label=label)
    #
    # plt.legend(loc='upper left')
    # plt.xlabel('Boosting Iterations')
    # plt.ylabel('Test Set Deviance')
    #
    # plt.show()
    # for c in cs:
    #     clf = svm.LinearSVC(C=c, dual=False)
    #     clf.fit(trainX, trainy)
    #     result = clf.predict(testX)
    #     error = np.count_nonzero(result - testy) / testX.shape[0]
    #     sensi, speci = my_scores(testy, result)
    #     # print error
    #     errors.append(error)
    #     sensis.append(sensi)
    #     specis.append(speci)
    # return errors, sensis, specis


def main():
    featuren = 1406
    key = str(featuren)
    dir_path = dir_path_dict[key]

    # key = '2900'
    key = '2900+850+556'
    data_str = dir_path + data_str_dict[key]
    # output_path = "SVM-on-feature-{}-2-fold.txt".format(key)
    n_estimators = [1000, 2000, 3000, 4000, 4500]
    # n_estimators = [3000]
    # n_estimators = [2000, 1000]
    for n in n_estimators:
        print 'n_estimators:\t{}'.format(n)
        errors = []
        sensis = []
        specis = []
        for i in range(10):
            data_path = data_str.format(i + 1)
            print data_path
            trainset, testset = get_dataset(data_path=data_path, foldi=i + 1, featuren=featuren)
            error, sensi, speci = basic_rf(trainset, testset, n_estimators=n)
            errors.append(error)
            sensis.append(sensi)
            specis.append(speci)
        print errors
        print "mean:\t{}\tstd:\t{}".format(np.mean(errors), np.std(errors))
        print "sensi:\t{}".format(np.mean(sensis))
        print "speci:\t{}".format(np.mean(specis))


if __name__ == '__main__':
    # times = 9
    # data_path = "feature1406-2-{}.pkl"
    #
    # data_tmp = "../exp4/feature1406-2-{}-shuffle_output.pkl"
    # loop_test(data_tmp=data_tmp)
    main()
    print "#####DONE#####"
