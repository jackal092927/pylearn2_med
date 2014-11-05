from pylearn2.expr.preprocessing import global_contrast_normalize

__author__ = 'Jackal'
import numpy as np
import cPickle
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial

class CIN_FEATURE2(DenseDesignMatrix):
    def __init__(self,
                 which_set,
                 datapath=None,
                 center=True,
                 rescale=True,
                 gcn=True):
        self.class_name = ['neg', 'pos']
        # load data
        path = "${PYLEARN2_DATA_PATH}/cin/"
        #datapath = path + 'feature850-2-1.pkl'
        if datapath is None:
            datapath = path + 'feature1406-2-1.pkl'
        datapath = serial.preprocess(datapath)
        with  open(datapath, 'rb') as f:
            #f = open(datapath, 'rb')
            train_set, valid_set, test_set = cPickle.load(f)
            #f.close()

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        if which_set == 'train':
            X, Y = self.train_set
        elif which_set == 'valid':
            X, Y = self.valid_set
        else:
            X, Y = self.test_set


        axis = 0
        _max = np.max(X, axis=axis)
        _min = np.min(X, axis=axis)
        _mean = np.mean(X, axis=axis)
        _std = np.std(X, axis=axis)
        _scale = _max - _min


        # print _max
        # print _min
        # print _mean
        # print _std

        if gcn:
            X = global_contrast_normalize(X, scale=gcn)
        else:
            if center:
                X[:, ] -= _mean
            if rescale:
                X[:, ] /= _scale


        X.astype(float)
        topo_view = X.reshape(X.shape[0], X.shape[1], 1, 1)
        # y = np.reshape(Y, (Y.shape[0], 1))
        y = np.atleast_2d(Y).T
        print X.shape
        print y.shape
        # print y.shape
        # y = np.zeros((Y.shape[0], 2))
        # y[:, 0] = Y
        # y[:, 0] = 1 - Y
        # super(CIN_FEATURE2, self).__init__(X=X, y=y)
        super(CIN_FEATURE2, self).__init__(topo_view=topo_view, y=y, y_labels=2)



CIN_FEATURE2("train")



