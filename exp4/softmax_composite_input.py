__author__ = 'Jackal'

from pylearn2.models.mlp import Softmax, Layer
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils import wraps
from theano.compat.python2x import OrderedDict
from theano import config
from theano.ifelse import ifelse
import theano.tensor as T
import numpy as np

class Softmax_composite_input(Softmax):

    @wraps(Layer.get_weights)
    def get_weights(self):
        if (not isinstance(self.input_space, VectorSpace)) and \
                (not isinstance(self.input_space, CompositeSpace)):
            raise NotImplementedError()

        return self.W.get_value()


    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        rval = OrderedDict()

        if not self.no_affine:
            W = self.W

            assert W.ndim == 2

            # sq_W = T.sqr(W)
            #
            # row_norms = T.sqrt(sq_W.sum(axis=1))
            # col_norms = T.sqrt(sq_W.sum(axis=0))
            #
            # rval.update(OrderedDict([('row_norms_min', row_norms.min()),
            #                          ('row_norms_mean', row_norms.mean()),
            #                          ('row_norms_max', row_norms.max()),
            #                          ('col_norms_min', col_norms.min()),
            #                          ('col_norms_mean', col_norms.mean()),
            #                          ('col_norms_max', col_norms.max()), ]))

        if (state_below is not None) or (state is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=1)

            rval.update(OrderedDict([('mean_max_class', mx.mean()),
                                     ('max_max_class', mx.max()),
                                     ('min_max_class', mx.min())]))

            if targets is not None:
                y_hat = T.argmax(state, axis=1)
                y = T.argmax(targets, axis=1)
                misclass = T.neq(y, y_hat).mean()
                misclass = T.cast(misclass, config.floatX)
                rval['misclass'] = misclass
                rval['nll'] = self.cost(Y_hat=state, Y=targets)

                # y_hat, y = self.convert2class(state, y)
                # misclass2 = T.neq(y, y_hat).mean()
                # misclass2 = T.cast(misclass2, config.floatX)
                # rval['misclass2'] = misclass2

                sensi, speci = self.get_sensi_speci(y_hat, y)
                rval['sensi'] = T.cast(sensi, config.floatX)
                rval['speci'] = T.cast(speci, config.floatX)

        return rval

    def convert2class(self, y_hat, y):
        # y_hat = T.set_subtensor(y_hat[(y_hat < 1).nonzero()], 0)
        # y_hat = T.set_subtensor(y_hat[(y_hat >= 1).nonzero()], 1)
        # y_hat = T.stacklists([y_hat[:, 0] + y_hat[:, 1], y_hat[:, 2] + y_hat[:, 3] + y_hat[:, 4]])
        y_hat = T.stacklists([T.sum(y_hat[:, 0:2], axis=1), T.sum(y_hat[:, 2:], axis=1)]).T
        y_hat = T.argmax(y_hat, axis=1)
        # y_hat = T.set_subtensor(y_hat[(y_hat < 2).nonzero()], 0)
        # y_hat = T.set_subtensor(y_hat[(y_hat >= 2).nonzero()], 1)
        y = T.set_subtensor(y[(y < 2).nonzero()], 0)
        y = T.set_subtensor(y[(y >= 2).nonzero()], 1)



        return [y_hat, y]


    def get_sensi_speci(self, y_hat, y):

        tag = 10 * y_hat + y

        tneg = T.cast((T.shape(tag[(T.eq(tag, 0.)).nonzero()]))[0], config.floatX)
        fneg = T.cast((T.shape(tag[(T.eq(tag, 1.)).nonzero()]))[0], config.floatX)
        fpos = T.cast((T.shape(tag[(T.eq(tag, 10.)).nonzero()]))[0], config.floatX)
        tpos = T.cast((T.shape(tag[(T.eq(tag, 11.)).nonzero()]))[0], config.floatX)
        # assert fneg + fneg + fpos + tpos == 1380
        # tneg.astype(config.floatX)
        # fneg.astype(config.floatX)
        # fpos.astype(config.floatX)
        # tpos.astype(config.floatX)

        speci = ifelse(T.eq((tneg + fpos), 0), np.float64('inf'), tneg / (tneg + fpos))
        sensi = ifelse(T.eq((tpos + fneg), 0), np.float64('inf'), tpos / (tpos + fneg))

        # keng die!!!
        # if T.eq((tneg + fpos), 0):
        #     speci = float('inf')
        # else:
        #     speci = tneg // (tneg + fpos)
        # if T.eq((tpos + fneg), 0):
        #     sensi = float('inf')
        # else:
        #     sensi = tpos // (tpos + fneg)

        # speci.astype(config.floatX)
        # sensi.astype(config.floatX)

        return [sensi, speci]


