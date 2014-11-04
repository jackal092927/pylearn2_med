__author__ = 'Jackal'

import os
from pylearn2.config import yaml_parse
from pylearn2.scripts.print_monitor import print_monitor

best_result_file = "sr_bgd_best.pkl"

datafile = 'sr0.yaml'
with open(datafile, 'r') as f:
    train = f.read()
hyper_params = {'n_classes' : 2,
                'nvis' : 850,
                'batch_size' : 1140,
                'best_result_file' : best_result_file
                }
train = train % (hyper_params)
# print train

train = yaml_parse.load(train)
train.main_loop()

#print_monitor(best_result_file)


