from __future__ import division
import matplotlib.pyplot as plt

__author__ = 'Jackal'
from sklearn.svm import l1_min_c
from datasets.cin_feature2_composite import CIN_FEATURE2
from sklearn import svm, linear_model
import numpy as np



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

def test_linearsvc(data_path="feature850-2-1.pkl"):
    train_set = CIN_FEATURE2(which_set='train', specs=False, data_path=data_path)
    test_set = CIN_FEATURE2(which_set='test', specs=False, data_path=data_path)
    X, y = train_set.get_raw_data()
    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)
    print cs
    errors = []
    for c in cs:
        clf = svm.LinearSVC(penalty='l1', C=c, dual=False)
        clf.fit(X, y)
        test_X, test_y = test_set.get_raw_data()
        result = clf.predict(test_X)
        error = np.count_nonzero(result - test_y) / test_X.shape[0]
        print error
        errors.append(error)

    return errors

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

if __name__ == '__main__':
    times = 9
    data_path = "feature1406-2-{}.pkl"
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # kernels = ['linear']
    # for kernel in kernels:
    #     results = cross_valid(times, kernel, datapath)
    #     print "kernel={}".format(kernel)
    #     print results
    #     print np.mean(results)
    #     print np.std(results)
    #
    # test_lr()

    # data_path = data_path.format(1)
    # print test_svm(data_path=data_path)
    errors = cross_valid(times=times, data_path=data_path)
    print errors
    print np.mean(errors)
    print np.std(errors)

