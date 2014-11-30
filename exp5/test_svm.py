from __future__ import division
import cPickle
import matplotlib.pyplot as plt
from pylearn2.expr.preprocessing import global_contrast_normalize
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

__author__ = 'Jackal'
from sklearn.svm import l1_min_c, SVC, LinearSVC
from datasets.cin_feature2_composite import *
from sklearn import svm, linear_model
import numpy as np
from sklearn.metrics import *

dir_path_dict = {'850' : "../results/mlp-1700-wd.0005-on-feature850-2-fold/",
                 '1406': "../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/",
                 '2086': "../results/mlpws-1700-1200-700-wd0.0005-on-feature2086-2/"}
data_str_dict = {'1700': "feature1700-850-2-fold{}_output.pkl",
                 '2550': "feature1700+850-2-fold{}_output.pkl",
                 '3600': "feature3600-2086-2-fold{}_output.pkl"}
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
    kernel = "poly"
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
    hyper_params = [{'C': cs, 'dual': [False], 'penalty':penalty}]
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



def get_dataset(data_path=None, foldi=1, featuren=1406):
    if data_path:
        with open(data_path, 'rb') as f:
            data = cPickle.load(f)
        (X, y), (test_X, test_y) = data
        y = np.argmax(y, axis=1)
        test_y = np.argmax(test_y, axis=1)
        return (X, y), (test_X, test_y)



    # train_set = CIN_FEATURE1406_2(which_set='train', specs=False, foldi=foldi)
    # valid_set = CIN_FEATURE1406_2(which_set='valid', specs=False, foldi=foldi)
    # test_set = CIN_FEATURE1406_2(which_set='test', specs=False, foldi=foldi)
    train_set = get_CIN_FEATURE(featuren=featuren, which_set='train', specs=False, foldi=foldi)
    valid_set = get_CIN_FEATURE(featuren=featuren, which_set='valid', specs=False, foldi=foldi)
    test_set = get_CIN_FEATURE(featuren=featuren, which_set='test', specs=False, foldi=foldi)
    testX, testy = test_set.get_raw_data()
    tdataset = zip(train_set.get_raw_data(), valid_set.get_raw_data())
    trainX = np.vstack(tdataset[0])
    trainy = np.hstack(tdataset[1])
    # trainX, trainy = [np.vstack(dataset) for dataset in tdataset]
    return ((trainX, trainy), (testX, testy))


def svc(trainset,
        testset,
        cs=[1]):
    kernel ='rbf'
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

def my_scores(y_hat, y):
    tag = 10 * y_hat + y
    t = np.bincount(tag)
    tn, fn, fp, tp = t[np.nonzero(t)]
    sensi = float(tp) / (tp + fn)
    speci = float(tn) / (tn + fp)
    return sensi, speci



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
        s = str(i+1)
        data_path = data_tmp.format(s)
        print data_path
        linsvm_cv(data_path=data_path)


def main():
    featuren = 850
    key = str(featuren)
    dir_path = dir_path_dict[key]
    key = str(featuren*2)
    data_str = dir_path + data_str_dict[key]
    output_path = "SVM_rbf-on-feature-{}-2-fold.txt".format(str(featuren*2))
    with open(output_path, 'a') as f:
        cs = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.1, 1, 10]#, 0.05, 0.5, 1, 10, 100, 200, 500, 1000, 1200, 1500]
        cs = [220, 230, 240, 250, 270]
        cs = [0.0005, 0.005, 0.001, 0.01, 0.1, 1, 3, 5, 10, 50, 100, 150, 200]
        cs = [130, 140, 160, 180, 300, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
        for c in cs:
            errors = []
            sensis = []
            specis = []
            msg = "###c={}".format(c)
            print msg
            f.write(msg+'\n')
            for i in range(10):
                data_path = data_str.format(i+1)
                trainset, testset = get_dataset(data_path=data_path, foldi=i+1, featuren=featuren)
                error, sensi, speci = svc(trainset, testset, cs=[c])
                errors.extend(error)
                sensis.extend(sensi)
                specis.extend(speci)
            print errors
            f.write(str(errors)+'\n')
            print np.mean(errors), np.mean(sensi), np.mean(specis), np.std(errors)
            f.write(str([np.mean(errors), np.mean(sensi), np.mean(specis), np.std(errors)])+'\n')

    # print np.mean(errors)

if __name__ == '__main__':
    # times = 9
    # data_path = "feature1406-2-{}.pkl"
    #
    # data_tmp = "../exp4/feature1406-2-{}-shuffle_output.pkl"
    # loop_test(data_tmp=data_tmp)
    main()
    print "#####DONE#####"
