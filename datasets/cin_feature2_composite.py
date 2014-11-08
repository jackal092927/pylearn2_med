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

        # def features_map_fn(indexes):
        #     rval = []
        #     for sequence_index, example_index in self._fetch_index(indexes):
        #         rval.append(self.samples_sequences[sequence_index][example_index:example_index
        #                                                                          + self.frames_per_example].ravel())
        #     return rval
        #
        # def targets_map_fn(indexes):
        #     rval = []
        #     for sequence_index, example_index in self._fetch_index(indexes):
        #         rval.append(self.samples_sequences[sequence_index][example_index
        #                                                            + self.frames_per_example].ravel())
        #     return rval

        # map_fn_components = [features_map_fn, targets_map_fn]
        # self.map_functions = tuple(map_fn_components)
        # self.cumulative_example_indexes = X.shape[0]

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
        y[:, 1] = 1 - Y
        print X.shape, y.shape
        super(CIN_FEATURE2, self).__init__(X=X, y=y)
        # super(CIN_FEATURE2, self).__init__(topo_view=topo_view, y=y, y_labels=2)

        if specs:
            assert X.shape[1] == (850 + 556)
            self.init_data_specs()
            self.feature850 = X[:, 0:850]
            self.feature556 = X[:, 850:]
            self.y = y


    # def _fetch_index(self, indexes):
    #     digit = np.digitize(indexes, self.cumulative_example_indexes) - 1
    #     return zip(digit,
    #                np.array(indexes) - self.cumulative_example_indexes[digit])
    #
    # def _validate_source(self, source):
    #     """
    #     Verify that all sources in the source tuple are provided by the
    #     dataset. Raise an error if some requested source is not available.
    #
    #     Parameters
    #     ----------
    #     source : `tuple` of `str`
    #         Requested sources
    #     """
    #     for s in source:
    #         try:
    #             self.data_specs[1].index(s)
    #         except ValueError:
    #             raise ValueError("the requested source named '" + s + "' " +
    #                              "is not provided by the dataset")

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

    # def get(self, source, indexes):
    #     """
    #     .. todo::
    #
    #         WRITEME
    #     """
    #     if type(indexes) is slice:  # eg: [3:5]
    #         indexes = np.arange(indexes.start, indexes.stop)
    #     self._validate_source(source)
    #     rval = []
    #     for so in source:
    #         batch = self.map_functions[self.data_specs[1].index(so)](indexes)
    #         batch_buffer = self.batch_buffers[self.data_specs[1].index(so)]
    #         dim = self.data_specs[0].components[self.data_specs[1].index(so)].dim
    #         if batch_buffer is None or batch_buffer.shape != (len(batch), dim):
    #             batch_buffer = np.zeros((len(batch), dim),
    #                                        dtype=batch[0].dtype)
    #         for i, example in enumerate(batch):
    #             batch_buffer[i] = example
    #         rval.append(batch_buffer)
    #     return tuple(rval)

    def get_data(self):

        if self.specs:
            return (self.feature850, self.feature556, self.y)
        else:
            return (self.X, self.y)

    # @functools.wraps(Dataset.iterator)
    # def iterator(self, mode=None, batch_size=None, num_batches=None,
    #              rng=None, data_specs=None, return_tuple=False):
    #     """
    #     .. todo::
    #
    #         WRITEME
    #     """
    #     if data_specs is None:
    #         data_specs = self._iter_data_specs
    #
    #     # If there is a view_converter, we have to use it to convert
    #     # the stored data for "features" into one that the iterator
    #     # can return.
    #     space, source = data_specs
    #     if isinstance(space, CompositeSpace):
    #         sub_spaces = space.components
    #         sub_sources = source
    #     else:
    #         sub_spaces = (space,)
    #         sub_sources = (source,)
    #
    #     convert = []
    #     for sp, src in safe_zip(sub_spaces, sub_sources):
    #         convert.append(None)
    #
    #     # TODO: Refactor
    #     if mode is None:
    #         if hasattr(self, '_iter_subset_class'):
    #             mode = self._iter_subset_class
    #         else:
    #             raise ValueError('iteration mode not provided and no default '
    #                              'mode set for %s' % str(self))
    #     else:
    #         mode = resolve_iterator_class(mode)
    #
    #     if batch_size is None:
    #         batch_size = getattr(self, '_iter_batch_size', None)
    #     if num_batches is None:
    #         num_batches = getattr(self, '_iter_num_batches', None)
    #     if rng is None and mode.stochastic:
    #         rng = self.rng
    #     return FiniteDatasetIterator(self,
    #                                  mode(self.num_examples, batch_size,
    #                                       num_batches, rng),
    #                                  data_specs=data_specs,
    #                                  return_tuple=return_tuple,
    #                                  convert=convert)

if __name__ == '__main__':
    CIN_FEATURE2("train")



