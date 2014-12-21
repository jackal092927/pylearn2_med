from __future__ import division
from sklearn.datasets import make_classification

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sys
from bayes_opt import BayesianOptimization
from utils.loader import *
import random

# Load data set and target values
# X_train, y_train = make_classification(n_samples=2500,
#                                    n_features=45,
#                                    n_informative=12,
#                                    n_redundant=7)

def load_data(i):

    dir_key, data_key = '1406', '850+556'
    dir_path = dir_path_dict[dir_key]
    data_str = dir_path + data_str_dict[data_key]
    data_path = data_str.format(i)
    print '#' * 53
    print data_path
    trainset, testset = get_dataset(data_path=data_path)
    return trainset, testset

def main():
    # stdout_path = 'outcome_testBO.txt'
    # print '[INFO]  stdout_path:\t{}'.format(stdout_path)
    # sys.stdout = open(stdout_path, 'w')
    #
    # np.random.seed(1)
    print '#' * 53
    scores = []
    sensis = []
    specis = []
    for i in range(10):
        trainset, testset = load_data(i + 1)
        X_train, y_train = trainset
        X_test, y_test = testset

        def svccv(C, tol):
            return cross_val_score(SVC(C=C, random_state=1, tol=tol),
                                   X_train, y_train, cv=9).mean()


        def rfccv(n_estimators, min_samples_split, max_features):
            return cross_val_score(RFC(n_estimators=int(n_estimators),
                                       min_samples_split=int(min_samples_split),
                                       max_features=min(max_features, 0.999),
                                       random_state=2),
                                   X_train, y_train, 'f1', cv=5).mean()

        svcBO = BayesianOptimization(svccv, {'C': (10, 50000), 'tol': (0.0001, 0.1)})
        svcBO.explore({'C': [10, 100, 1000, 10000, 20000, 50000], 'tol': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]})

        # rfcBO = BayesianOptimization(rfccv, {'n_estimators': (10, 250),
        # 'min_samples_split': (2, 25),
        # 'max_features': (0.1, 0.999)})

        svcBO.maximize(init_points=50, restarts=200, n_iter=100)

        print '#' * 53
        print 'Final Results'
        print 'SVC: %f' % svcBO.res['max']['max_val']
        print 'max_params: ', svcBO.res['max']['max_params']

        params = svcBO.res['max']['max_params']
        clf = SVC(C=params['C'], random_state=1, tol=params['tol'])
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        result = clf.predict(X_test)
        sensi, speci = my_scores(y_test, result)
        print 1 - score, sensi, speci
        # print 'err:', 1 - score

        scores.append(score)
        sensis.append(sensi)
        specis.append(speci)

    print scores
    print "accur:\t{}\tstd:\t{}".format(np.mean(scores), np.std(scores))
    print "sensi:\t{}".format(np.mean(sensis))
    print "speci:\t{}".format(np.mean(specis))

if __name__ == "__main__":
    errors = []
    svcs = []
    with open("outcome_testBO.txt", 'r') as f:
        for line in f.readlines():
            if line.startswith("err: "):
                err = float(line[4:])
                errors.append(err)
            if line.startswith("SVC: "):
                svcs.append(1-float(line[4:]))


    print "svcs: ", svcs
    print np.mean(svcs), np.std(svcs)
    print "errors: ", errors
    print np.mean(errors), np.std(errors)




