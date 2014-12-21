#!/usr/bin/env python
import cPickle
import os
from os.path import isfile
import subprocess
import tarfile
import datetime
import time


__author__ = 'Jackal'

import sys
from pylearn2.utils import serial
from pylearn2.gui import get_weights_report
import numpy as np
import gc
# from exp4 import dir_path_dict
from utils.loader import *


# keys = ["test_y_misclass", "train_y_misclass", "valid_y_misclass", "test_y_misclass2", "test_y_sensi", "test_y_speci"]
keys = ["test_y_misclass", "train_y_misclass", "valid_y_misclass", "test_y_sensi", "test_y_speci"]
# dir_path = "../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/"
# dir_path = "../results/softmax-wd.0005/"


n_fold = 10
def my_monitor(models=None, n_fold=9, model_str="mlp3-1700-1700-on-feature850-2-{}.pkl"):
    if models is None:
        models = []
        for i in range(1, n_fold+1):
            models.append(model_str.format(str(i)))

    errors = []
    for model_path in models:
        # if len(models) > 1:
        print model_path
        model = serial.load(model_path)
        monitor = model.monitor
        if not hasattr(monitor, '_epochs_seen'):
            print 'old file, not all fields parsed correctly'
        else:
            print 'epochs seen: ', monitor._epochs_seen
            print 'time trained: ', max(channels[key].time_record[-1] for key in
                                        channels)
        del model
        gc.collect()
        channels = monitor.channels
        keys = ["test_y_misclass"]#, "train_y_misclass", "valid_y_misclass"]
        for key in keys:
            value = channels[key].val_record[-1]
            print key, ':', value
            errors.append(value)

    return errors

def my_show_weights(model_path, rescale='individual', border=False, out=None):
    pv = get_weights_report.get_weights_report(model_path=model_path,
                                               rescale=rescale,
                                               border=border)

    if out is None:
        pv.show()
    else:
        pv.save(out)




def single_print_gz(filepath):
    errors = {}
    assert tarfile.is_tarfile(filepath)
    post = '_' + str(long(time.time()))
    path = filepath[:-4]  # remove .tgz
    dname = os.path.join(os.path.dirname(path), post)
    fname = os.path.basename(path)
    path = os.path.join(dname, fname)
    # cmd = "tar -xzvf {} -C {} {}".format(filepath, dname, fname)
    # print cmd
    # subprocess.call(cmd, shell=True)
    tar = tarfile.open(filepath)
    tar.extractall(path=dname)
    tar.close()
    assert isfile(path)

    model = serial.load(path)
    monitor = model.monitor
    del model
    gc.collect()
    channels = monitor.channels
    for key in keys:
        value = channels[key].val_record[-1]
        if isinstance(value, np.ndarray):
            value = value.min()
        print key, ':', value
        errors[key] = value

    cmd = 'rm -rf {}'.format(dname)
    print cmd
    subprocess.call(cmd, shell=True)
    return errors


def single_print(filepath):
    errors = {}
    assert os.path.isfile(filepath)
    model = serial.load(filepath)
    monitor = model.monitor
    del model
    gc.collect()
    channels = monitor.channels
    for key in keys:
        value = channels[key].val_record[-1]
        if isinstance(value, np.ndarray):
            value = value.min()
        print key, ':', value
        errors[key] = value
    return errors

def model_print(strmodel, nfold=10):

    results = {}
    for key in keys:
        results.setdefault(key, [])
    # errors = []

    for i in range(nfold):
        filepath = strmodel.format(i+1)
        print filepath
        errors = single_print_gz(filepath)
        for key in keys:
            value = errors[key]
            results[key] += [value]
            # results[key] = results.get(key, []) + [value]

    for key in keys:
        result = results[key]
        print key
        print result
        print np.mean(result)
        print np.std(result)

    # for result in results:
    #     for key, value in result:
    #         if "test_y_misclass" == key:
    #             errors.append(value)
    # print errors
    # print np.mean(errors)
    # print np.std(errors)


def saveaspkl(data, path):
    with open(path, 'wb') as f:
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)



def get_options():
    # feature_n = 'mlp-'
    dir_key = 'mlp-single'
    model_str = dir_path_dict[dir_key] + "mlp-1200-1200-wd0.0005-on-feature556-2-fold{}.pkl.tgz"

    # model_str = dir_path_dict[dir_name] + "mlp-1700-1200-wd0.0005-on-feature855+556-2-fold{}.pkl"
    nfold = 10

    if len(sys.argv) >= 3:
        model_str = sys.argv[1]
        nfold = int(sys.argv[2])
    return model_str, nfold


def print_mlp_single_556(dim_h0=1200, dir_key='mlp-single'):
    dim_h1s = [1200, 900, 300]
    featuren = 556
    for dim_h1 in dim_h1s:
        dir_path = dir_path_dict[dir_key]
        data_path = "mlp-{}.{}-wd0.0005-on-feature{}-2-fold{}.pkl.tgz". \
            format(dim_h0, dim_h1, featuren, '{}')
        report_path = dir_path + "report_mlp-{}.{}-wd0.0005-on-feature{}-2.txt".format(dim_h0, dim_h1, featuren)
        # print '[INFO]  stdout_path:\t{}'.format(report_path)
        sys.stdout = open(report_path, 'w')
        strmodel = dir_path + data_path

        model_print(strmodel=strmodel)


def print_mlp_single_680(dim_h0=1400, dir_key='mlp-single'):
    dim_h1s = [1400, 1000, 400]
    featuren = 680
    for dim_h1 in dim_h1s:
        dir_path = dir_path_dict[dir_key]
        data_path = "mlp-{}.{}-wd0.0005-on-feature{}-2-fold{}.pkl.tgz". \
            format(dim_h0, dim_h1, featuren, '{}')
        report_path = dir_path + "report_mlp-{}.{}-wd0.0005-on-feature{}-2.txt".format(dim_h0, dim_h1, featuren)
        # print '[INFO]  stdout_path:\t{}'.format(report_path)
        sys.stdout = open(report_path, 'w')
        strmodel = dir_path + data_path

        model_print(strmodel=strmodel)



def main():
    # strmodel, nfold = get_options()
    # filepath = strmodel.format(1)
    # filepath = "../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/mlp-1700-1200-wd.0005.0005.0-on-feature850+556-2-fold10.pkl"
    # single_print(filepath=filepath)
    # filepath = "../results/mlp-1700-1200-wd.0005-on-feature1406-2-fold/mlp-1700-1200-wd0.0005-on-feature855+556-2-fold10.pkl.tgz"
    # single_print_gz(filepath=filepath)
    print_mlp_single_680()


if __name__ == '__main__':
    main()



