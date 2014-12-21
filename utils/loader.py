from __future__ import division
import os
import subprocess
import time
from os.path import isfile
from datasets.cin_feature2_composite import *
import numpy as np
import gzip
from os import path
import pickle
import tarfile

# key: original input feature size
dir_path_dict = {'850': "../results/mlp-1700-wd.0005-on-feature850-2-fold/",
                 '1406': "../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/",
                 '2086': "../results/mlpws-1700-1200-700-wd0.0005-on-feature2086-2/",
                 'softmax': "../results/softmax-wd.0005/",
                 'mlp-single': "../results/mlp-single/"}
# key: feature size (or 850+556)
data_str_dict = {'1700': "feature1700-850-2-fold{}_output.pkl.tgz",
                 '1700+850': "feature1700+850-2-fold{}_output.pkl.tgz",
                 '2900': "feature2900-1406-2-fold{}_output.pkl.tgz",
                 '1406a': "feature2900-1406a-2-fold{}_output.pkl.gz",
                 '2900+1406': "feature2900+1406-2-fold{}_output.pkl.tgz",
                 '3600': "feature3600-2086-2-fold{}_output.pkl.tgz",
                 '850+556': "feature2900-850+556-2-fold{}_output.pkl.tgz",
                 '2900+850+556': "feature2900+850+556-2-fold{}_output.pkl.tgz",
                 '850+556a': "feature2900-850+556a-2-fold{}_output.pkl.gz"}
# key: original input feature size
model_str_dict = {'850': "mlp-1700-wd0.0005-on-feature850-2-fold{}.pkl.tgz",
                  '556': "mlp-1200-1200-wd0.0005-on-feature556-2-fold{}.pkl.tgz",
                  '680': "mlp-1400-1400-wd0.0005-on-feature680-2-fold{}.pkl.tgz",
                  '1406': "mlp-1700-1200-wd0.0005-on-feature1406-2-fold{}.pkl.tgz",
                  '2086':  "mlpws-1700-1200-700-wd0.0005-on-feature2086-2-{}.pkl.tgz",
                  '850+556': "mlp-1700-1200-wd0.0005-on-feature855+556-2-fold{}.pkl.tgz"}
                  # '850+556': "mlp-1700-1200-wd0.0005-on-feature850+556-2-fold{}.pkl.gz"}

_data_str_dict = {'1700': dir_path_dict['850'] +"feature1700-850-2-fold{}_output.pkl.tgz",
                 '1700+850': dir_path_dict['850'] +"feature1700+850-2-fold{}_output.pkl.tgz",
                 '2900': dir_path_dict['1406'] +"feature2900-1406a-2-fold{}_output.pkl.gz",
                 # '1406a': dir_path_dict['1406'] +"feature2900-1406a-2-fold{}_output.pkl.gz",
                 '2900+1406': dir_path_dict['1406'] +"feature2900+1406a-2-fold{}_output.pkl.tgz",
                 '3600': dir_path_dict['2086'] +"feature3600-2086-2-fold{}_output.pkl.tgz",
                 '850+556': dir_path_dict['1406'] +"feature2900-850+556-2-fold{}_output.pkl.tgz",
                 '2900+850+556': dir_path_dict['1406'] +"feature2900+850+556-2-fold{}_output.pkl.tgz"}
                 # '850+556a': dir_path_dict[1406] +"feature2900-850+556a-2-fold{}_output.pkl.gz"}
# key: original input feature size
_model_str_dict = {'850': dir_path_dict['850'] +"mlp-1700-wd0.0005-on-feature850-2-fold{}.pkl.tgz",
                  '556': dir_path_dict['softmax'] +"mlp-1200-1200-wd0.0005-on-feature556-2-fold{}.pkl.tgz",
                  '680': dir_path_dict['softmax'] +"mlp-1400-1400-wd0.0005-on-feature680-2-fold{}.pkl.tgz",
                  '1406': dir_path_dict['1406'] +"mlp-1700-1200-wd0.0005-on-feature1406-2-fold{}.pkl.tgz",
                  '2086': dir_path_dict['2086'] +"mlpws-1700-1200-700-wd0.0005-on-feature2086-2-{}.pkl.tgz",
                  '850+556': dir_path_dict['1406'] +"mlp-1700-1200-wd0.0005-on-feature855+556-2-fold{}.pkl.tgz"}
# '850+556': "mlp-1700-1200-wd0.0005-on-feature850+556-2-fold{}.pkl.gz"}




def datapath_helper(key):
    results = []
    for i in range(10):
        results.append(_data_str_dict[key].format(i + 1))
    return results

def modelpath_helper(key):
    results = []
    for i in range(10):
        results.append(_model_str_dict[key].format(i + 1))
    return results

def check_path_exist(filepath):
    if path.isfile(filepath):
        filepath += '_' + str(int(time.time()))
    return filepath

def saveas_pkl(data, filepath, overlap=False):
    if not overlap: filepath = check_path_exist(filepath)
    with open(filepath, 'wb') as f:
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)


def saveas_pkl_gz(data, filepath, overlap=False):
    if not overlap: filepath = check_path_exist(filepath)
    with gzip.open(filepath, 'wb') as f:
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)

def pkl2gz(pkl_path, gz_path, overlap=False):
    f_in = open(pkl_path, 'rb')
    f_out = gzip.open(gz_path, 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()


## load data from .pkl or .gz
def safeload(data_path):
    if data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            data = cPickle.load(f)
    elif data_path.endswith('.tgz'):
        tar = tarfile.open(data_path)
        f = tar.extractfile(tar.next())
        data = cPickle.load(f)
        f.close()
    else: # data_path.endswith('.gz'):
        with gzip.open(data_path, 'rb') as f:
            data = cPickle.load(f)

    return data

def myload_model(filepath):
    assert tarfile.is_tarfile(filepath)
    post = '_' + str(long(time.time()))
    path_ = filepath[:-4]  # remove .tgz
    dname = os.path.join(os.path.dirname(path_), post)
    fname = os.path.basename(path_)
    path_ = os.path.join(dname, fname)
    # cmd = "tar -xzvf {} -C {} {}".format(filepath, dname, fname)
    # print cmd
    # subprocess.call(cmd, shell=True)
    tar = tarfile.open(filepath)
    tar.extractall(path=dname)
    tar.close()
    assert isfile(path_)

    model = serial.load(path_)

    cmd = 'rm -rf {}'.format(dname)
    print cmd
    subprocess.call(cmd, shell=True)
    return model


def get_dataset(data_path=None, foldi=1, featuren=None):
    if data_path:
        data = safeload(data_path)
        (X, y), (test_X, test_y) = data
        # TODO
        # y = np.argmax(y, axis=1)
        # test_y = np.argmax(test_y, axis=1)
        y = y[:, 0]
        test_y = test_y[:, 0]
        return (X, y), (test_X, test_y)

    # train_set = CIN_FEATURE1406_2(which_set='train', specs=False, foldi=foldi)
    # valid_set = CIN_FEATURE1406_2(which_set='valid', specs=False, foldi=foldi)
    # test_set = CIN_FEATURE1406_2(which_set='test', specs=False, foldi=foldi)
    train_set = get_CIN_FEATURE(featuren=featuren, which_set='train', specs=False, foldi=foldi)
    valid_set = get_CIN_FEATURE(featuren=featuren, which_set='valid', specs=False, foldi=foldi)
    test_set = get_CIN_FEATURE(featuren=featuren, which_set='test', specs=False, foldi=foldi)
    testX, testy = test_set.get_raw_data()
    tdataset = zip(train_set.get_raw_data(), valid_set.get_raw_data())
    trainX = np.vstack(tdataset[0])
    trainy = np.hstack(tdataset[1])
    # trainX, trainy = [np.vstack(dataset) for dataset in tdataset]
    return ((trainX, trainy), (testX, testy))


def my_scores(y_hat, y):
    tag = 10 * y_hat + y
    tag = tag.astype(int)
    t = np.bincount(tag)
    tn, fn, fp, tp = t[np.nonzero(t)]
    sensi = float(tp) / (tp + fn)
    speci = float(tn) / (tn + fp)
    return sensi, speci


def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)

    return L