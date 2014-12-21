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
from datasets import filestr_dic, dirpath, feature_ns



def get_CIN_FEATURE(featuren,
                    which_set,
                    data_path=None,
                    center=True,
                    rescale=True,
                    gcn=True,
                    specs=True,
                    foldi=1,
                    foldn=10):
    if featuren == 850: return CIN_FEATURE850_2(which_set=which_set,
                                                specs=specs,
                                                foldi=foldi)
    if featuren == 1406: return CIN_FEATURE1406_2(which_set=which_set,
                                                  specs=specs,
                                                  foldi=foldi)
    if featuren == 2086: return CIN_FEATURE2086_2(which_set=which_set,
                                                  specs=specs,
                                                  foldi=foldi)

class CIN_FEATURE2(DenseDesignMatrix):
    def __init__(self,
                 which_set,
                 data_path=None,
                 center=True,
                 rescale=True,
                 gcn=True,
                 specs=True):
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
        self.specs = specs
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
        self.raw_X = X
        self.raw_y = Y
        y = np.zeros((Y.shape[0], 2))
        y[:, 0] = Y
        y[:, 1] = 1 - Y
        # print "Load CIN_FEATURE2 data: {}, with size X:{}, y:{}".format(data_path, X.shape, y.shape)
        super(CIN_FEATURE2, self).__init__(X=X, y=y)
        # super(CIN_FEATURE2, self).__init__(topo_view=topo_view, y=y, y_labels=2)

        if specs:
            assert X.shape[1] == (850 + 556)
            self.init_data_specs()
            self.feature850 = X[:, 0:850]
            self.feature556 = X[:, 850:]
            self.y = y

    def init_data_specs(self):

        self.data_specs = \
            (
                CompositeSpace
                (
                    [
                        # CompositeSpace([VectorSpace(dim=850),VectorSpace(dim=556)]),
                        VectorSpace(dim=850),
                        VectorSpace(dim=556),
                        VectorSpace(dim=2)
                    ]
                ),
                (
                    'feature850', 'feature556', 'targets'
                )
            )

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `TIMIT.get()`
            _does_ respect those data specs.
        """
        return self.data_specs


    def get_data(self):

        if self.specs:
            return (self.feature850, self.feature556, self.y)
        else:
            return (self.X, self.y)


    def get_raw_data(self):
        return self.raw_X, self.raw_y


class CIN_FEATURE(DenseDesignMatrix):

    def __init__(self,
                 which_set,
                 center=True,
                 rescale=True,
                 gcn=True,
                 specs=False,
                 foldi=1,
                 foldn=10):
        self.class_name = ['neg', 'pos']
        self.specs = specs
        self.gcn = gcn
        self.which_set = which_set
        self.foldi = foldi
        self.foldn = foldn

        self.set_filestr()
        self.load()


    def set_filestr(self):
        # self.filestr = filestr_dic['1406']
        raise NotImplementedError()

    def load(self):
        foldn = self.foldn
        foldi = self.foldi

        # load data
        if self.which_set == 'valid':
            i = (foldi) % foldn
            print "valid fold: {}".format(i+1)
            filepath = self.filestr.format(str(i + 1))
            filepath = dirpath + filepath
            filepath = serial.preprocess(filepath)
            X, Y = self.loadi(filepath)
        elif self.which_set == 'test':
            i = (foldi - 1) % foldn
            print "test fold: {}".format(foldi)
            filepath = self.filestr.format(str(i + 1))
            filepath = dirpath + filepath
            filepath = serial.preprocess(filepath)
            X, Y = self.loadi(filepath)
        else:
            print "train folds"
            indexs = range(foldn)
            i = foldi % foldn
            indexs.pop(i)
            if i == 0:
                indexs.pop(-1)
            else:
                i = (foldi - 1) % foldn
                indexs.pop(i)
            Xs = []
            Ys = []
            for i in indexs:
                filepath = self.filestr.format(str(i + 1))
                filepath = dirpath + filepath
                filepath = serial.preprocess(filepath)
                X, Y = self.loadi(filepath)
                Xs.append(X)
                Ys.append(Y)
            X = np.vstack(Xs)
            Y = np.hstack(Ys)

        # print X.shape, Y.shape

        X.astype(float)

        if self.gcn:
            X = self._gcn(X)
        # else:
        # if center:
        #         X[:, ] -= _mean
        #     if rescale:
        #         X[:, ] /= _scale


        # topo_view = X.reshape(X.shape[0], X.shape[1], 1, 1)
        # y = np.reshape(Y, (Y.shape[0], 1))
        # y = np.atleast_2d(Y).T
        self.raw_X = X
        self.raw_y = Y
        y = np.zeros((Y.shape[0], 2))
        y[:, 0] = Y
        y[:, 1] = 1 - Y
        # print "Load CIN_FEATURE1406_2 data: {}, with size X:{}, y:{}".format(data_path, X.shape, y.shape)
        super(CIN_FEATURE, self).__init__(X=X, y=y)
        # super(CIN_FEATURE2, self).__init__(topo_view=topo_view, y=y, y_labels=2)

        if self.specs:
            self.set_specs()

    def set_specs(self):
        # assert X.shape[1] == (850 + 556)
        # self.init_data_specs()
        # self.feature850 = X[:, 0:850]
        # self.feature556 = X[:, 850:850 + 556]
        # self.data = [self.feature850, self.feature556, self.y]
        # self.feature680 = X[:, 850 + 556:]
        # self.y = y
        raise NotImplementedError()


    def init_data_specs(self):
        sps = [ VectorSpace(dim=data.shape[1]) for data in self.data]
        tags = ['feature' + str(data.shape[1]) for data in self.data[:-1]]
        tags.append('targets')
        tags = tuple(tags)
        self.data_specs = (CompositeSpace(sps), tags)
        # print self.data_specs

        # self.data_specs = \
        # (
        #         CompositeSpace
        #         (
        #             [
        #                 VectorSpace(dim=850),
        #                 VectorSpace(dim=556),
        #                 VectorSpace(dim=2)
        #             ]
        #         ),
        #         (
        #             'feature850',
        #             'feature556',
        #             'targets'
        #         )
        # )

    def _gcn(self, X):
        if not self.specs:
            X = global_contrast_normalize(X, scale=True)
            return X
        pre = 0
        for n in feature_ns:
            if n <= X.shape[1]:
                X[:, pre:n] = global_contrast_normalize(X[:, pre:n], scale=True)
                pre = n
        return X
        #
        # X[:, :850] = global_contrast_normalize(X[:, :850], scale=True)
        # X[:, 850:] = global_contrast_normalize(X[:, 850:], scale=True)
        # raise NotImplementedError()


    def loadall(self):
        datasets = []
        for i in range(self.foldn):
            filename = self.filestr.format(str(i + 1))
            filename = dirpath + filename
            filename = serial.preprocess(filename)
            print "load data file: " + filename
            self.loadi(i, filename=filename)

        dataset = datasets[0]
        X, y = datasetXy
        # print X.shape, y.shape

        return datasets

    def loadi(self,
              filename):
        with open(filename, 'rb') as f:
            print "load file: " + filename
            datasetXy = cPickle.load(f)
        return datasetXy



    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `TIMIT.get()`
            _does_ respect those data specs.
        """
        return self.data_specs


    def get_data(self):

        if self.specs:
            # return (self.feature850, self.feature556, self.feature680, self.y)
            return self.data
        else:
            return (self.X, self.y)


    def get_raw_data(self):
        return self.raw_X, self.raw_y


class CIN_FEATURE850_2(CIN_FEATURE):
    feature_n = 850

    def __init__(self,
                 which_set,
                 # center=True,
                 # rescale=True,
                 gcn=True,
                 specs=False,
                 foldi=1
                 # foldn=10
    ):
        super(CIN_FEATURE850_2, self).__init__(
            which_set=which_set, gcn=gcn, specs=specs, foldi=foldi)

    def set_filestr(self):
        self.filestr = filestr_dic['850']

    def set_specs(self):
        pass


class CIN_FEATURE556_2(CIN_FEATURE):
    feature_n = '556'

    def __init__(self,
                 which_set,
                 # center=True,
                 # rescale=True,
                 gcn=True,
                 specs=False,
                 foldi=1
                 # foldn=10
    ):
        super(CIN_FEATURE556_2, self).__init__(
            which_set=which_set, gcn=gcn, specs=specs, foldi=foldi)

    def set_filestr(self):
        self.filestr = filestr_dic[self.feature_n]

    def set_specs(self):
        pass



class CIN_FEATURE680_2(CIN_FEATURE):
    feature_n = '680'

    def __init__(self,
                 which_set,
                 # center=True,
                 # rescale=True,
                 gcn=True,
                 specs=False,
                 foldi=1
                 # foldn=10
    ):
        super(CIN_FEATURE680_2, self).__init__(
            which_set=which_set, gcn=gcn, specs=specs, foldi=foldi)

    def set_filestr(self):
        self.filestr = filestr_dic[self.feature_n]

    def set_specs(self):
        pass


class CIN_FEATURE1406_2(CIN_FEATURE):
    feature_n = "1406"

    def __init__(self,
                 which_set,
                 # center=True,
                 # rescale=True,
                 gcn=True,
                 specs=True,
                 foldi=1
                 # foldn=10
    ):
        super(CIN_FEATURE1406_2, self).__init__(
            which_set=which_set, gcn=gcn, specs=specs, foldi=foldi)

    def set_filestr(self):
        self.filestr = filestr_dic[self.feature_n]

    def set_specs(self):
        X = self.X
        assert X.shape[1] == (850 + 556)
        self.feature850 = X[:, 0:850]
        self.feature556 = X[:, 850:850 + 556]
        self.data = (self.feature850, self.feature556, self.y)
        # self.feature680 = X[:, 850 + 556:]
        # self.y = y
        self.init_data_specs()

    # def _gcn(self, X):
    #     X = global_contrast_normalize(X, scale=True)
    #     return X


class CIN_FEATURE2086_2(CIN_FEATURE):
    feature_n = "2086"

    def __init__(self,
                 which_set,
                 # center=True,
                 # rescale=True,
                 gcn=True,
                 specs=True,
                 foldi=1
                 # foldn=10
    ):
        super(CIN_FEATURE2086_2, self).__init__(
            which_set=which_set, gcn=gcn, specs=specs, foldi=foldi)

    def set_filestr(self):
        self.filestr = filestr_dic[self.feature_n]

    def set_specs(self):
        X = self.X
        assert X.shape[1] == (850 + 556 + 680)
        self.feature850 = X[:, 0:850]
        self.feature556 = X[:, 850:850 + 556]
        self.feature680 = X[:, 850 + 556:]
        self.data = (self.feature850, self.feature556, self.feature680, self.y)
        # self.y = y
        self.init_data_specs()



class CIN_FEATURE2086_5(DenseDesignMatrix):
    dirpath = "${PYLEARN2_DATA_PATH}/cin/"
    # filestr = "feature2086-2-{}.pkl"

    def __init__(self,
                 which_set,
                 data_path=None,
                 center=True,
                 rescale=True,
                 gcn=True,
                 specs=True,
                 foldi=1,
                 foldn=10,
                 filestr="feature2086-5-{}.pkl"):
        self.class_name = ['neg', 'cin1','cin2','cin3','cancer']
        # load data
        self.specs = specs
        self.filestr = filestr

        if which_set == 'valid':
            i = (foldi) % foldn
            filepath = self.filestr.format(str(i + 1))
            filepath = self.dirpath + filepath
            filepath = serial.preprocess(filepath)
            X, Y = self.loadi(filepath)
        elif which_set == 'test':
            i = (foldi - 1) % foldn
            filepath = self.filestr.format(str(i + 1))
            filepath = self.dirpath + filepath
            filepath = serial.preprocess(filepath)
            X, Y = self.loadi(filepath)
        else:
            indexs = range(foldn)
            i = foldi % foldn
            indexs.pop(i)
            if i == 0:
                indexs.pop(-1)
            else:
                i = (foldi - 1) % foldn
                indexs.pop(i)
            Xs = []
            Ys = []
            for i in indexs:
                filepath = self.filestr.format(str(i + 1))
                filepath = self.dirpath + filepath
                filepath = serial.preprocess(filepath)
                X, Y = self.loadi(filepath)
                Xs.append(X)
                Ys.append(Y)
            X = np.vstack(Xs)
            Y = np.hstack(Ys)

        print X.shape, Y.shape
        # col0s = np.where(Y == 0)[0]
        # print len(col0s)

        X.astype(float)
        axis = 0
        _max = np.max(X, axis=axis)
        _min = np.min(X, axis=axis)
        _mean = np.mean(X, axis=axis)
        _std = np.std(X, axis=axis)
        _scale = _max - _min

        if gcn:
            X[:, :850] = global_contrast_normalize(X[:, :850], scale=gcn)
            X[:, 850:850+556] = global_contrast_normalize(X[:, 850:850 + 556], scale=gcn)
            X[:, 850+556:] = global_contrast_normalize(X[:, 850+556:], scale=gcn)
        # else:
        #     if center:
        #         X[:, ] -= _mean
        #     if rescale:
        #         X[:, ] /= _scale


        # topo_view = X.reshape(X.shape[0], X.shape[1], 1, 1)
        # y = np.reshape(Y, (Y.shape[0], 1))
        # y = np.atleast_2d(Y).T
        self.raw_X = X
        self.raw_y = Y
        y = np.zeros((Y.shape[0], 5))
        for i in range(Y.shape[0]):
            j = Y[i]
            y[i, j] = 1
        # print y[:, :]
        # y[:, 0] = Y
        # y[:, 1] = 1 - Y
        print "Load CIN_FEATURE2086_5 data: {}, with size X:{}, y:{}".format(data_path, X.shape, y.shape)
        super(CIN_FEATURE2086_5, self).__init__(X=X, y=y)
        # super(CIN_FEATURE2, self).__init__(topo_view=topo_view, y=y, y_labels=2)

        if specs:
            assert X.shape[1] == (850 + 556 + 680)
            self.init_data_specs()
            self.feature850 = X[:, 0:850]
            self.feature556 = X[:, 850:850 + 556]
            self.feature680 = X[:, 850 + 556:]
            self.y = y

    def loadall(self,
                dirpath="${PYLEARN2_DATA_PATH}/cin/",
                filestr="feature2086-2-{}.pkl",
                n=10):
        datasets = []
        for i in range(n):
            filename = filestr.format(str(i + 1))
            filename = dirpath + filename
            filename = serial.preprocess(filename)
            print "load data file: " + filename
            self.loadi(i, filename=filename)

        dataset = datasets[0]
        X, y = datasetXy
        print X.shape, y.shape

        return datasets

    def loadi(self,
              filename):
        with open(filename, 'rb') as f:
            print "load file: " + filename
            datasetXy = cPickle.load(f)
        return datasetXy

    def init_data_specs(self):

        self.data_specs = \
            (
                CompositeSpace
                (
                    [
                        # CompositeSpace([VectorSpace(dim=850),VectorSpace(dim=556)]),
                        VectorSpace(dim=850),
                        VectorSpace(dim=556),
                        VectorSpace(dim=680),
                        VectorSpace(dim=5)
                    ]
                ),
                (
                    'feature850', 'feature556', 'feature680', 'targets'
                )
            )

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `TIMIT.get()`
            _does_ respect those data specs.
        """
        return self.data_specs


    def get_data(self):

        if self.specs:
            return (self.feature850, self.feature556, self.feature680, self.y)
        else:
            return (self.X, self.y)


    def get_raw_data(self):
        return self.raw_X, self.raw_y

if __name__ == '__main__':
    # CIN_FEATURE2("train")
    # feature850 = CIN_FEATURE850_2("train")
    # feature1406 = CIN_FEATURE1406_2("train")
    # feature2086 = CIN_FEATURE2086_2('train')
    # print feature2086.raw_X[:, :1406] - feature1406.raw_X
    data1 = CIN_FEATURE680_2('test', foldi=10)
    # data2 = CIN_FEATURE1406_2('train', foldi=10)
    data2 = CIN_FEATURE2086_2('test', foldi=10)
    print data1.raw_X - data2.raw_X[:, 1406:]
    # print data2.raw_y
    # print feature850.raw_X - feature1406.raw_X[:, 0:850]
    # print feature1406.raw_X[:, :850].shape







