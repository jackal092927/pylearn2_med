from datasets.cin_feature2_composite import *
from exp4.mlp_output import MLP_output

__author__ = 'Jackal'

from scipy import misc
from numpy import *
import matplotlib.pyplot as plt
import cPickle
import gzip
import scipy.io
import os
from pylearn2.utils import serial
from utils.loader import *

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

def cgzip(filepath):
    with open(filepath, 'rb') as fin:
        with gzip.open(filepath, 'wb') as fou:
            fou.writelines(fin)


def xgzip(filepath):
    with gzip.open(filepath, 'rb') as f:
        content = cPickle.load(f)

    return content

def cmp_dataset():

    # featuren = 1406
    # dir_key = '1406'
    # model_key = '1406'
    # data_key = '2900'
    i = 8
    # output = MLP_output(foldi=i + 1, featuren=featuren,
    #                     dir_key=dir_key, model_key=model_key, data_key=data_key)
    #
    # X = np.hstack(output.trainX)
    # y = output.trainy
    # print X.shape, y.shape

    trainset, testset = safeload(
        "../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900+850+556-2-fold10_output.pkl.tgz")
    trainset0, testset0 = safeload(
        "../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-850+556-2-fold10_output.pkl.tgz_1419037479")
    trainX, trainy = trainset
    trainX0, trainy0 = trainset0
    testX, testy = testset
    testX0, testy0 = testset0
    # print np.sum(trainy - y)
    # trainX = np.mean(trainX, axis=0)
    # trainX0 = np.mean(trainX0, axis=0)
    # print trainX.shape
    delta = trainX - trainX0
    # delta = delta[np.nonzero(delta)]
    # print delta.shape
    # delta = delta[:, 0:2900]
    # delta = delta[:, 1:]
    # delta = delta[:, 2800:]
    print delta
    print np.max(delta)
    print np.sum(delta)
    print np.mean(trainX), np.std(trainX)
    print np.mean(trainX0), np.std(trainX0)
    # cin = CIN_FEATURE1406_2('train', foldi=9)
    # print cin.raw_X - X[0:1104, :]
    # print cin.raw_y - y[0:1104]



def main():

    # datafile = '/Users/Jackal/data/pylearn/cifar10/cifar-10-batches-py/data_batch_1'
    # filepath = "../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/feature2900-855+556-2-fold1_output.pkl"
    # content = serial.load(filepath)
    # print content

    cmp_dataset()




if __name__ == '__main__':
    main()

