__author__ = 'Jackal'

from scipy import misc
from numpy import *
import matplotlib.pyplot as plt
import cPickle
import gzip
import scipy.io
import os


def cat_pkl(
        datafile):
    with open(datafile, 'rb') as f:
        set_list = cPickle.load(f)

    print set_list.nvis
    #
    # for set in set_list:
    #     for s in set:
    #         print s.shape


def cat_mat(
        datafile='/Users/Jackal/Work/theano/data/850invariPLBP-78.84/Neg.mat',
        dname='Neg'):
    # filename = os.path.basename(datapath)
    matdata = scipy.io.loadmat(datafile)
    data = matdata[dname]
    print data.shape

def cat_cifar(datafile):
    f = open(datafile, 'rb')
    dict = cPickle.load(f)
    f.close()
    return dict


# cat_mat()
#datafile = '/Users/Jackal/data/pylearn/cifar10/cifar-10-batches-py/data_batch_1'
datafile = './l1_grbm.pkl'
cat_pkl(datafile=datafile)
#
# print [v for v in iter(dict)]
# print dict['batch_label']
# print dict['filenames']
