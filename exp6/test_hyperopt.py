from __future__ import division
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import random_forest, standard_scaler, min_max_scaler, svc, svc_linear, any_classifier, any_preprocessing
from hyperopt import tpe
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.loader import *
import sys

RANDOM_SEED = 101

def test_hyperopt():
    # Load data
    featuren = 1406
    dir_key = '1406'
    data_key = '850+556'
    dir_path = dir_path_dict[dir_key]
    data_str = dir_path + data_str_dict[data_key]

    # stdout_path = 'outcome_hyperopt_svc.moreinfo1.txt'
    # print '[INFO]  stdout_path:\t{}'.format(stdout_path)
    # sys.stdout = open(stdout_path, 'w')

    print "[INFO]  params:\tclassifier=svc_linear('mySVC'), algo=tpe.suggest, preprocessing=[standard_scaler('std_scl')]"
    scores = []
    sensis = []
    specis = []
    for i in range(10):
        # Load data
        data_path = data_str.format(i + 1)
        print data_path
        trainset, testset = get_dataset(data_path=data_path, foldi=i + 1, featuren=featuren)
        train_data, train_label = trainset
        test_data, test_label = testset

        # Create the estimator object
        estim = hyperopt_estimator(classifier=any_classifier('mySVC'),
                                   algo=tpe.suggest,
                                   preprocessing=[standard_scaler('std_scl')],
                                   seed=RANDOM_SEED)

        # Search the space of classifiers and preprocessing steps and their
        # respective hyperparameters in sklearn to fit a model to the data
        estim.fit(train_data, train_label)

        # show instances of the best classifier
        model = estim.best_model()
        print model

        # Make a prediction using the optimized model
        prediction = estim.predict(test_data)
        error = np.count_nonzero(prediction - test_label) / test_data.shape[0]
        sensi, speci = my_scores(test_label, prediction)
        print 1 - error, sensi, speci

        # Report the accuracy of the classifier on a given set of data
        score = estim.score(test_data, test_label)
        print score

        scores.append(score)
        sensis.append(sensi)
        specis.append(speci)

    print scores
    print "accur:\t{}\tstd:\t{}".format(np.mean(scores), np.std(scores))
    print "sensi:\t{}".format(np.mean(sensis))
    print "speci:\t{}".format(np.mean(specis))

def hyperopt_850_556():
    # Load data
    dir_key = '1406'
    data_key = '850+556'
    dir_path = dir_path_dict[dir_key]
    data_str = dir_path + data_str_dict[data_key]

    # Redirect stdout to file
    stdout_path = 'outcome_hyperopt_any.any.txt'
    print '[INFO]  stdout_path:\t{}'.format(stdout_path)
    sys.stdout = open(stdout_path, 'w')
    print "[INFO]  params:\talgo=tpe.suggest"

    # Train
    scores = []
    sensis = []
    specis = []
    for i in range(10):
        # Load data
        data_path = data_str.format(i + 1)
        print data_path
        trainset, testset = get_dataset(data_path=data_path)
        train_data, train_label = trainset
        test_data, test_label = testset

        # Create the estimator object
        # estim = hyperopt_estimator(classifier=any_classifier('mySVC'),
        # algo=tpe.suggest,
        #                            preprocessing=[standard_scaler('std_scl')])
        estim = hyperopt_estimator(algo=tpe.suggest, seed=RANDOM_SEED)

        # Search the space of classifiers and preprocessing steps and their
        # respective hyperparameters in sklearn to fit a model to the data
        estim.fit(train_data, train_label)

        # show instances of the best classifier
        model = estim.best_model()
        print model

        # Make a prediction using the optimized model
        prediction = estim.predict(test_data)
        error = np.count_nonzero(prediction - test_label) / test_data.shape[0]
        sensi, speci = my_scores(test_label, prediction)
        print 1 - error, sensi, speci

        # Report the accuracy of the classifier on a given set of data
        score = estim.score(test_data, test_label)
        print score

        scores.append(score)
        sensis.append(sensi)
        specis.append(speci)

    print scores
    print "accur:\t{}\tstd:\t{}".format(np.mean(scores), np.std(scores))
    print "sensi:\t{}".format(np.mean(sensis))
    print "speci:\t{}".format(np.mean(specis))


def train850():
    # Load data
    dir_key = '850'
    data_key = '1700'
    dir_path = dir_path_dict[dir_key]
    data_str = dir_path + data_str_dict[data_key]

    stdout_path = 'outcome_hyperopt_svc-on-feature1700.txt'
    print '[INFO]  stdout_path:\t{}'.format(stdout_path)
    import sys

    sys.stdout = open(stdout_path, 'w')
    print "[INFO]  params:\tclassifier=svc_linear('mySVC'), algo=tpe.suggest, preprocessing=[standard_scaler('std_scl')]"
    scores = []
    sensis = []
    specis = []
    for i in range(10):
        # Load data
        data_path = data_str.format(i + 1)
        print data_path
        trainset, testset = get_dataset(data_path=data_path)
        train_data, train_label = trainset
        test_data, test_label = testset

        # Create the estimator object
        estim = hyperopt_estimator(classifier=svc_linear('mySVC'),
                                   algo=tpe.suggest,
                                   preprocessing=[standard_scaler('std_scl')],
                                   seed=RANDOM_SEED)

        # Search the space of classifiers and preprocessing steps and their
        # respective hyperparameters in sklearn to fit a model to the data
        estim.fit(train_data, train_label)

        # show instances of the best classifier
        model = estim.best_model()
        print model

        # Make a prediction using the optimized model
        prediction = estim.predict(test_data)
        error = np.count_nonzero(prediction - test_label) / test_data.shape[0]
        sensi, speci = my_scores(test_label, prediction)
        print 1.0 - error, sensi, speci

        # Report the accuracy of the classifier on a given set of data
        score = estim.score(test_data, test_label)
        print score

        scores.append(score)
        sensis.append(sensi)
        specis.append(speci)

    print scores
    print "accur:\t{}\tstd:\t{}".format(np.mean(scores), np.std(scores))
    print "sensi:\t{}".format(np.mean(sensis))
    print "speci:\t{}".format(np.mean(specis))




def main():
    # test_hyperopt()
    data_paths = []
    errs = []
    prep = []
    clfs = []

    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold1_output.pkl.tgz")
    errs.append(1-0.804347826087)
    clfs.append(SVC(C=996.276098557, cache_size=1000.0, class_weight=None, coef0=0.0,
                     degree=3, gamma=0.0, kernel='rbf', max_iter=88980969, probability=False,
                     random_state=1, shrinking=True, tol=0.0761796218413, verbose=False))
    prep.append(StandardScaler(copy=True, with_mean=False, with_std=False))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold2_output.pkl.tgz")
    errs.append(1-0.905797101449)
    clfs.append(SVC(C=22137.3884984, cache_size=1000.0, class_weight=None, coef0=0.0,
                    degree=3, gamma=0.0, kernel='rbf', max_iter=496138031, probability=False,
                    random_state=2, shrinking=True, tol=0.00213078045767, verbose=False))
    prep.append(StandardScaler(copy=True, with_mean=False, with_std=True))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold3_output.pkl.tgz")
    errs.append(1-0.884057971014)
    clfs.append(SVC(C=43525.8240707, cache_size=1000.0, class_weight=None, coef0=0.0,
                    degree=3, gamma=0.0, kernel='rbf', max_iter=755647886, probability=False,
                    random_state=0, shrinking=False, tol=0.00058584209437, verbose=False))
    prep.append(StandardScaler(copy=True, with_mean=False, with_std=True))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold4_output.pkl.tgz")
    errs.append(1-0.898550724638)
    clfs.append(SVC(C=9.98284800858, cache_size=1000.0, class_weight=None, coef0=0.0,
                    degree=3, gamma=0.0, kernel='rbf', max_iter=318725199, probability=False,
                    random_state=1, shrinking=True, tol=0.000421350777388, verbose=False))
    prep.append(StandardScaler(copy=True, with_mean=False, with_std=True))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold5_output.pkl.tgz")
    errs.append(1-0.891304347826)
    clfs.append(SVC(C=2017.04641626, cache_size=1000.0, class_weight=None,
                    coef0=0.0404595659137, degree=5.0, gamma=0.0, kernel='poly',
                    max_iter=14408316, probability=False, random_state=1, shrinking=False,
                    tol=0.00221763912474, verbose=False))
    prep.append (StandardScaler(copy=True, with_mean=True, with_std=True))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold6_output.pkl.tgz")
    errs.append(1-0.855072463768)
    clfs.append(SVC(C=1940.75249739, cache_size=1000.0, class_weight=None,
                    coef0=0.531231688696, degree=2.0, gamma=0.0, kernel='poly',
                    max_iter=114748544, probability=False, random_state=0, shrinking=False,
                    tol=0.00421763022456, verbose=False))
    prep.append (StandardScaler(copy=True, with_mean=True, with_std=True))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold7_output.pkl.tgz")
    errs.append(1-0.934782608696)
    clfs.append(SVC(C=38055.9387429, cache_size=1000.0, class_weight=None, coef0=0.0,
                    degree=3, gamma=0.2056652177, kernel='rbf', max_iter=24788283,
                    probability=False, random_state=2, shrinking=True, tol=3.85673222108e-08,
                    verbose=False))
    prep.append (StandardScaler(copy=True, with_mean=False, with_std=False))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold8_output.pkl.tgz")
    errs.append(1-0.920289855072)
    clfs.append(SVC(C=42.2299113575, cache_size=1000.0, class_weight=None,
                    coef0=0.633349078806, degree=5.0, gamma=0.0, kernel='poly',
                    max_iter=227262787, probability=False, random_state=3, shrinking=True,
                    tol=0.0055410706406, verbose=False))
    prep.append (StandardScaler(copy=True, with_mean=True, with_std=True))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold9_output.pkl.tgz")
    errs.append(1-0.847826086957)
    clfs.append(SVC(C=13163037.3637, cache_size=1000.0, class_weight=None,
                    coef0=0.978923171314, degree=3.0, gamma=0.0, kernel='poly',
                    max_iter=830083848, probability=False, random_state=1, shrinking=True,
                    tol=0.000195487054627, verbose=False))
    prep.append(StandardScaler(copy=True, with_mean=True, with_std=True))
    data_paths.append("../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold10_output.pkl.tgz")
    errs.append(1-0.876811594203)
    clfs.append(SVC(C=1036.44020761, cache_size=1000.0, class_weight=None,
                    coef0=0.804921086483, degree=2.0, gamma=3.4661676473, kernel='poly',
                    max_iter=583986675, probability=False, random_state=3, shrinking=False,
                    tol=0.00185258643189, verbose=False))
    prep.append(StandardScaler(copy=True, with_mean=True, with_std=False))

    train_errs = []
    test_errs = []



    for i in range(1):
        # Load data
        trainset, testset = get_dataset(data_path=data_paths[i])
        X_train, y_train = trainset
        X_test, y_test = testset

        # Preprocess
        pre = prep[i]
        X_train = pre.fit_transform(X_train)
        X_test = pre.fit_transform(X_test)

        C = 1000
        tol = 0.00000001

        # epoch = 0
        # while True:
        # epoch += 1
        # print '#' * 53
        # print "EPOCH: ", epoch

        # Classifier
        # clf = SVC(C=C, tol=tol,
        #           cache_size=1000.0, class_weight=None, coef0=0.0,
        #           degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
        #           random_state=1, shrinking=True, verbose=False)
        # clf = SVC(C=22137.3884984, cache_size=1000.0, class_weight=None, coef0=0.0,
        #           degree=3, gamma=0.0, kernel='rbf', max_iter=496138031, probability=False,
        #           random_state=2, shrinking=True, tol=0.00213078045767, verbose=False)
        clf = SVC(C=30000, cache_size=1000.0, class_weight=None, coef0=0.0,
            degree=3, gamma=0.0, kernel='poly', max_iter=-1, probability=False,
            random_state=1, shrinking=True, tol=0.0005, verbose=False)
        clf = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                  gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
                  random_state=None, shrinking=True, tol=0.001, verbose=False)
        # clf = clfs[i]
        print clf

        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        test_errs.append(test_acc)
        train_errs.append(train_acc)
        print "{} ==> {}".format(train_acc, test_acc)
        # if train_acc == 1:
        #     return
        # else:
        #     C *= 2
        return

    print train_errs
    print test_errs
    print np.mean(test_errs)
    print [1-err for err in errs]




if __name__ == '__main__':
    main()
    # print np.mean([0.78260869565217395, 0.87681159420289856, 0.86956521739130432, 0.86956521739130432, 0.89855072463768115, 0.8623188405797102,
    # 0.78985507246376807, 0.92028985507246375, 0.85507246376811596, 0.76811594202898548])
    # test_hyperopt()

