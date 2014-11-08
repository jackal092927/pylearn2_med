__author__ = 'Jackal'
import numpy as np
import cPickle
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial, safe_zip
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.space import CompositeSpace, VectorSpace
from utils.iteration import FiniteDatasetIterator

import functools
from pylearn2.datasets import Dataset


class CIN_FEATURE2(DenseDesignMatrix):
    def __init__(self,
                 which_set,
                 data_path=None,
                 center=True,
                 rescale=True,
                 gcn=True):
        self.class_name = ['neg', 'pos']
        # load data
        path = "${PYLEARN2_DATA_PATH}/cin/"
        #datapath = path + 'feature850-2-1.pkl'
        if data_path is None:
            data_path = path + 'feature1406-2-1.pkl'
        else:
            data_path = path + data_path
        data_path = serial.preprocess(data_path)
        with  open(data_path, 'rb') as f:
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

        X.astype(float)
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

        # topo_view = X.reshape(X.shape[0], X.shape[1], 1, 1)
        # y = np.reshape(Y, (Y.shape[0], 1))
        # y = np.atleast_2d(Y).T
        y = np.zeros((Y.shape[0], 2))
        y[:, 0] = Y
        y[:, 0] = 1 - Y
        print X.shape, y.shape
        super(CIN_FEATURE2, self).__init__(X=X, y=y)
        # super(CIN_FEATURE2, self).__init__(topo_view=topo_view, y=y, y_labels=2)

if __name__ == '__main__':
    CIN_FEATURE2("train")



