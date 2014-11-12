from theano.scalar import float32

__author__ = 'Jackal'

#from pylearn2.datasets.mnist import MNIST
from pylearn2.config import yaml_parse
import numpy as np
from theano import config
import theano.tensor as T
from theano import function
from theano.ifelse import ifelse


# X = T.dvector('X')
# y = T.set_subtensor(X[(X <= 1).nonzero()], 0)
# y = T.set_subtensor(y[(y > 1).nonzero()], 1)
# y = y[(T.eq(y, 0)).nonzero()]
#
# y = T.shape(y)
#
# arr = np.array([1., 2.0, 3.0, 0, 1, 2, 1, 2, 4, 3, 2, 2])
# arr.astype(np.float64)
# print y.eval({X: arr})
#



y_hat = T.dvector('y_hat')
y = T.dvector('y')
# results = []
# for i in [0, 1, 10, 11]:
#     t = T.max(T.shape(tag[(T.eq(tag, i)).nonzero()]))
#     # t = T.eq(t, 2)
#     print t.eval({tag: arr})
#     # results.append(T.shape(t))
# # print results

def convert2class(y_hat, y):
    y_hat = T.set_subtensor(y_hat[(y_hat < 1).nonzero()], 0)
    y_hat = T.set_subtensor(y_hat[(y_hat >= 1).nonzero()], 1)
    y = T.set_subtensor(y[(y < 1).nonzero()], 0)
    y = T.set_subtensor(y[(y >= 1).nonzero()], 1)
    # for i, v in enumerate(y):
    # if v <= 1:
    #         y[i] = 0
    #     else:
    #         y[i] = 1
    #     for i, v in enumerate(y_hat):
    #         if v <= 1:
    #             y_hat[i] = 0
    #         else:
    #             y_hat[i] = 1

    return y_hat, y



arr1 = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1])
arr2 = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])


def get_sensi_speci(y_hat, y):
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

    speci = ifelse(T.eq((tneg + fpos), 0), np.float64(float('inf')), tneg / (tneg + fpos))
    sensi = ifelse(T.eq((tpos + fneg), 0), np.float64(float('inf')), tpos / (tpos + fneg))

    # keng die!!!
    # if T.eq((tneg + fpos), 0):
    #     speci = float('inf')
    # else:
    #     speci = tneg // (tneg + fpos)
    # if T.eq((tpos + fneg), 0.):
    #     sensi = float('inf')
    # else:
    #     sensi = tpos // (tpos + fneg)

    # speci.astype(config.floatX)
    # sensi.astype(config.floatX)
    return [sensi, speci]


# y_hat2, y2 = convert2class(y_hat, y)
# misclass2 = T.neq(y2, y_hat2).mean()
# misclass2 = T.cast(misclass2, config.floatX)
# test = T.mean(T.neq(y, y_hat))
# f = function(inputs=[y_hat, y], outputs=misclass2)
# print f(arr1, arr2)

tag = 10 * y_hat + y
tneg = T.cast(T.max(T.shape(tag[(T.eq(tag, 0.)).nonzero()])), config.floatX)
fneg = T.cast(T.max(T.shape(tag[(T.eq(tag, 1.)).nonzero()])), config.floatX)
fpos = T.cast(T.max(T.shape(tag[(T.eq(tag, 10.)).nonzero()])), config.floatX)
tpos = T.cast(T.max(T.shape(tag[(T.eq(tag, 11.)).nonzero()])), config.floatX)
sensi = tneg / (tneg + fpos)
f = function(inputs=[y_hat, y], outputs=sensi)
print f(arr1, arr2)
g = function(inputs=[y_hat, y], outputs=get_sensi_speci(y_hat, y))


print g(arr1, arr2)

