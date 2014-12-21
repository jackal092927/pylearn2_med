from __future__ import division
import cPickle
import gzip
import matplotlib.pyplot as plt
from pylearn2.expr.preprocessing import global_contrast_normalize
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

__author__ = 'Jackal'
from sklearn.svm import l1_min_c, SVC, LinearSVC
from datasets.cin_feature2_composite import *
from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import *

from utils.loader import *


def test_svm(data_path):
    kernel = "linear"
    gamma = 1 / 32768
    C = 1020
    probability = True
    train_set = CIN_FEATURE2(which_set='train', specs=False, data_path=data_path)
    test_set = CIN_FEATURE2(which_set='test', specs=False, data_path=data_path)
    X, y = train_set.get_raw_data()
    # y = np.array(y)
    # print X.shape, y.shape

    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, probability=probability)
    clf.fit(X, y)
    test_X, test_y = test_set.get_raw_data()
    result = clf.predict(test_X)
    n = np.count_nonzero(result - test_y)
    error = n / test_X.shape[0]
    print error
    return error

def svm_cv(data_path):
    kernel = "rbf"
    gamma = 1 / 32768
    C = 1020
    probability = True
    # train_set = CIN_FEATURE2(which_set='train', specs=False, data_path=data_path)
    # test_set = CIN_FEATURE2(which_set='test', specs=False, data_path=data_path)
    # X, y = train_set.get_raw_data()
    # test_X, test_y = test_set.get_raw_data()
    with open(data_path, 'rb') as f:
        data = cPickle.load(f)
    (X, y), (test_X, test_y) = data

    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 5)
    cs = [1., 10., 100., C]
    cs = [0.0005, 0.001, 1, 10, 100, C, 1500]
    gammas = [0.001, 0.0001, gamma]
    hyper_params = [{'kernel': [kernel], 'C': cs,
                     'gamma': gammas, 'probability': [probability]}]
    scores = ['precision', 'recall']
    scores = [None]

    for score in scores:
        # print("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(SVC(C=1), hyper_params, cv=5)
        clf.fit(X, y)
        print("Best parameters set found on development set:")
        print(clf.best_estimator_)
        print("Grid scores on development set:")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = test_y, clf.predict(test_X)
        print(classification_report(y_true, y_pred))
        n = np.count_nonzero(y_pred - test_y)
        error = n / test_X.shape[0]
        print "error rate:\t", error


        # clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, probability=probability)
        # clf.fit(X, y)
        # test_X, test_y = test_set.get_raw_data()
        # result = clf.predict(test_X)
        # n = np.count_nonzero(result - test_y)
        # error = n / test_X.shape[0]
        # print error
        # return error


def rbfsvm_cv(trainset, testset, scaler=None):
    trainx, trainy = trainset
    testx, testy = testset
    # trainx = trainx[:, 2900+850:]
    # testx = testx[:, 2900+850:]
    if scaler:
        trainx = scaler.fit_transform(trainx)
        trainy = scaler.fit_transform(trainy)
    cs = 10.0 ** np.arange(-5, 5)
    gs = 10.0 ** np.arange(-5, 0)
    ks = ['rbf']
    params = [{'kernel': ks, 'C': cs,
               'gamma': gs}]
    clf = GridSearchCV(SVC(), param_grid=params, cv=9)
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


def rbfsvm_cv2(trainset, testset, scaler=None):
    trainx, trainy = trainset
    testx, testy = testset
    # trainx = trainx[:, 2900+850:]
    # testx = testx[:, 2900+850:]
    if scaler:
        trainx = scaler.fit_transform(trainx)
        trainy = scaler.fit_transform(trainy)
    # cs = 2.0 ** np.arange(6, 7)
    cs = [100.0]
    gs = 2.0 ** np.arange(-4, 1)
    ks = ['rbf']
    params = [{'kernel': ks, 'C': cs,
               'gamma': gs}]
    clf = GridSearchCV(SVC(), param_grid=params, cv=9)
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

def linsvm_cv(data_path):
    C = 1020
    probability = True
    # train_set = CIN_FEATURE2(which_set='train', specs=False, data_path=data_path)
    # test_set = CIN_FEATURE2(which_set='test', specs=False, data_path=data_path)
    # X, y = train_set.get_raw_data()
    # test_X, test_y = test_set.get_raw_data()
    with open(data_path, 'rb') as f:
        data = cPickle.load(f)
    gcn = True
    (X, y), (test_X, test_y) = data
    X = global_contrast_normalize(X, scale=gcn)
    test_X = global_contrast_normalize(test_X, scale=gcn)

    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 5)
    cs = [0.0005, 0.001, 1, 10, 100, C, 1500]
    penalty = ['l2']
    hyper_params = [{'C': cs, 'dual': [False], 'penalty': penalty}]
    # scores = ['precision']#, 'recall']
    scores = [None]
    for score in scores:
        # print("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(LinearSVC(C=1), hyper_params, cv=5)
        clf.fit(X, y)
        print("Best parameters set found on development set:")
        print(clf.best_estimator_)
        print("Grid scores on development set:")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = test_y, clf.predict(test_X)
        print(classification_report(y_true, y_pred))
        n = np.count_nonzero(y_pred - test_y)
        error = n / test_X.shape[0]
        print "error rate:\t", error


        # clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, probability=probability)
        # clf.fit(X, y)
        # test_X, test_y = test_set.get_raw_data()
        # result = clf.predict(test_X)
        # n = np.count_nonzero(result - test_y)
        # error = n / test_X.shape[0]
        # print error
        # return error

def svc(trainset,
        testset,
        cs=[1]):
    kernel = 'rbf'
    gamma = 0.0005
    trainX, trainy = trainset
    testX, testy = testset
    errors = []
    sensis = []
    specis = []
    for c in cs:
        clf = svm.SVC(kernel=kernel, C=c, gamma=gamma)
        clf.fit(trainX, trainy)
        result = clf.predict(testX)
        error = np.count_nonzero(result - testy) / testX.shape[0]
        sensi, speci = my_scores(testy, result)
        # print error
        errors.append(error)
        sensis.append(sensi)
        specis.append(speci)
    return errors, sensis, specis

def linearsvc(trainset,
              testset,
              cs=[1]):
    # cs = l1_min_c(trainX, trainy, loss='log') * np.logspace(0, 3)
    # print cs
    trainX, trainy = trainset
    testX, testy = testset
    errors = []
    sensis = []
    specis = []
    for c in cs:
        clf = svm.LinearSVC(C=c, dual=False)
        clf.fit(trainX, trainy)
        result = clf.predict(testX)
        error = np.count_nonzero(result - testy) / testX.shape[0]
        sensi, speci = my_scores(testy, result)
        # print error
        errors.append(error)
        sensis.append(sensi)
        specis.append(speci)
    return errors, sensis, specis

def cross_valid(times, data_path):
    # savepath = "./mlp4_{}.{}-on-{}"
    result = []
    for i in range(times):
        data_path = data_path.format(str(i + 1))
        #print data_path
        result.append(test_svm(data_path=data_path))
    return result

def test_lr(data_path="feature850-2-1.pkl"):
    train_set = CIN_FEATURE2(which_set='train', specs=False, data_path=data_path)
    test_set = CIN_FEATURE2(which_set='test', specs=False, data_path=data_path)
    X, y = train_set.get_raw_data()
    test_X, test_y = test_set.get_raw_data()
    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)
    print cs
    return
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, y)
        result = clf.predict(test_X)
        n = np.count_nonzero(result - test_y)

        print n / test_X.shape[0]
    # print coefs_
    # clf.fit(X, y)

    coefs_ = np.array(coefs_)

    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')
    plt.axis('tight')
    plt.show()
    # result = clf.predict(test_X)


    # return float(n) / test_X.shape[0]


def loop_test(data_tmp):
    for i in range(10):
        s = str(i + 1)
        data_path = data_tmp.format(s)
        print data_path
        linsvm_cv(data_path=data_path)


def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)

    return L


def test_rbfsvm_cv():
    # dir_key = '1406'
    # data_key = '850+556'
    # dir_path = dir_path_dict[dir_key]
    # data_str = dir_path + data_str_dict[data_key]
    key = '2900+850+556'
    datapaths = datapath_helper(key)
    scaler = StandardScaler()
    scaler = None
    # print datapaths
    train_scos = []
    test_scos = []
    for i in range(10):
        datapath = datapaths[i]
        print '#' * 53
        print datapath
        trainset, testset = get_dataset(data_path=datapath)
        train_sco, test_sco = rbfsvm_cv2(trainset, testset, scaler)
        train_scos.append(train_sco)
        test_scos.append(test_sco)
    print "train_scores:"
    print train_scos
    print "mean:\t{} std:\t{}".format(np.mean(train_scos), np.std(train_scos))
    print "test_scores:"
    print test_scos
    print "mean:\t{} std:\t{}".format(np.mean(test_scos), np.std(test_scos))


def polsvm_cv(trainset, testset, scaler=None):
    trainx, trainy = trainset
    testx, testy = testset
    if scaler:
        trainx = scaler.fit_transform(trainx)
        trainy = scaler.fit_transform(trainy)
    cs = 10.0 ** np.arange(-5, 6)
    gs = 10.0 ** np.arange(-5, 2)
    degs = [2,3,4,5]
    coefs = 0.1 * np.arange(0, 10)
    tols = 10.0 ** np.arange(-5, 0)
    ks = ['poly']
    params = [{'kernel': ks,
               'C': cs,
               'gamma': gs,
               'coef0': coefs,
               'degree': degs,
               'tol': tols}]
    clf = GridSearchCV(SVC(), param_grid=params, cv=9)
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


def test_polsvm_cv():
    # key = '850+556'
    key = '1700'
    datapaths = datapath_helper(key)
    scaler = StandardScaler()
    # scaler = None
    # print datapaths
    train_scos = []
    test_scos = []
    for i in range(10):
        datapath = datapaths[i]
        print '#' * 53
        print datapath
        trainset, testset = get_dataset(data_path=datapath)
        train_sco, test_sco = polsvm_cv(trainset, testset, scaler)
        train_scos.append(train_sco)
        test_scos.append(test_sco)
    print "train_scores:"
    print train_scos
    print "mean:\t{} std:\t{}".format(np.mean(train_scos), np.std(train_scos))
    print "test_scores:"
    print test_scos
    print "mean:\t{} std:\t{}".format(np.mean(test_scos), np.std(test_scos))


def main():
    featuren = 1406
    key = str(featuren)
    dir_path = dir_path_dict[key]
    key = '1406a'
    data_str = dir_path + data_str_dict[key]
    output_path = "SVM-on-feature-{}-2-fold.txt".format('2900')
    with open(output_path, 'a') as f:
        cs = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.1, 1,
              10]  # , 0.05, 0.5, 1, 10, 100, 200, 500, 1000, 1200, 1500]
        # cs = [220, 230, 240, 250, 270]
        # cs = [0.0005, 0.005, 0.001, 0.01, 0.1, 1, 3, 5, 10, 50, 100, 150, 200]
        # cs = [130, 140, 160, 180, 300, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
        # cs = [1200]
        cs = frange(0.1, 1, 0.005)
        cs = [0.007]
        for c in cs:
            errors = []
            sensis = []
            specis = []
            msg = "###c={}".format(c)
            print msg
            f.write(msg + '\n')
            for i in range(10):
                data_path = data_str.format(i + 1)
                trainset, testset = get_dataset(data_path=data_path, foldi=i + 1, featuren=featuren)
                error, sensi, speci = linearsvc(trainset, testset, cs=[c])
                errors.extend(error)
                sensis.extend(sensi)
                specis.extend(speci)
            print errors
            f.write(str(errors) + '\n')
            print np.mean(errors), np.mean(sensi), np.mean(specis), np.std(errors)
            f.write(str([np.mean(errors), np.mean(sensi), np.mean(specis), np.std(errors)]) + '\n')

            # print np.mean(errors)

np.random.seed(1)
def shuffle_dataset(x, y):
    # trainx, trainy = trainset
    # testx, testy = testset
    y = np.atleast_2d(y).T
    t = np.hstack([x, y])
    np.random.shuffle(t)
    x, y = t[:, :-1], t[:, -1]
    # y = list(y)
    return x, y



def valid_svm(shuffle=True):
    key = '850+556'
    datapaths = datapath_helper(key)
    train_scos = []
    test_scos = []
    sensis = []
    specis = []
    clf = SVC(C=100, gamma=0.1, kernel='rbf')
    print clf
    for i in range(10):
        datapath = datapaths[i]
        print '#' * 53
        print datapath
        trainset, testset = get_dataset(data_path=datapath)
        trainx, trainy = trainset
        testx, testy = testset
        if shuffle:
            trainx, trainy = shuffle_dataset(trainx, trainy)
            print trainy
            testx, testy = shuffle_dataset(testx, testy)
        clf.fit(trainx, trainy)
        res = clf.predict(testx)
        sensi, speci = my_scores(res, testy)
        trainacc = clf.score(trainx, trainy)
        print "train acc:\t", trainacc
        testacc = clf.score(testx, testy)
        print "test acc: ", testacc
        train_scos.append(trainacc)
        test_scos.append(testacc)
        sensis.append(sensi)
        specis.append(speci)
        # train_sco, test_sco = rbfsvm_cv(trainset, testset, scaler)
        # train_scos.append(train_sco)
        # test_scos.append(test_sco)
    print "train_scores:"
    print train_scos
    print "mean:\t{} std:\t{}".format(np.mean(train_scos), np.std(train_scos))
    print "test_scores:"
    print test_scos
    print "mean:\t{} std:\t{}".format(np.mean(test_scos), np.std(test_scos))
    print "sensi:\t{}".format(np.mean(sensis))
    print "speci:\t{}".format(np.mean(specis))


def valid_svm2():
    key = '2900'
    datapaths = datapath_helper(key)
    train_scos = []
    test_scos = []
    sensis = []
    specis = []
    clf = SVC(C=1000, gamma=0.1, kernel='rbf')
    print clf
    for i in range(10):
        datapath = datapaths[i]
        print '#' * 53
        print datapath
        trainset, testset = get_dataset(data_path=datapath)
        trainx, trainy = trainset
        testx, testy = testset
        # trainx = trainx[:, 2900:]
        # testx = testx[:, 2900:]
        clf.fit(trainx, trainy)
        res = clf.predict(testx)
        sensi, speci = my_scores(res, testy)
        trainacc = clf.score(trainx, trainy)
        print "train acc:\t", trainacc
        testacc = clf.score(testx, testy)
        print "test acc: ", testacc
        train_scos.append(trainacc)
        test_scos.append(testacc)
        sensis.append(sensi)
        specis.append(speci)
        # train_sco, test_sco = rbfsvm_cv(trainset, testset, scaler)
        # train_scos.append(train_sco)
        # test_scos.append(test_sco)
    print "train_scores:"
    print train_scos
    print "mean:\t{} std:\t{}".format(np.mean(train_scos), np.std(train_scos))
    print "test_scores:"
    print test_scos
    print "mean:\t{} std:\t{}".format(np.mean(test_scos), np.std(test_scos))
    print "sensi:\t{}".format(np.mean(sensis))
    print "speci:\t{}".format(np.mean(specis))

if __name__ == '__main__':
    # outcome_rbfsvm_cv_stdscl0-on-feature2900+850+556.txt

    # main()
    # test_rbfsvm_cv()
    valid_svm2()
    print "#####DONE#####"


