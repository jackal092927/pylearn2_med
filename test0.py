__author__ = 'Jackal'

#from pylearn2.datasets.mnist import MNIST
from pylearn2.config import yaml_parse


layer1_yaml = open('l1_grbm.yaml', 'r').read()
hyper_params_l1 = {'batch_size': 20,
                   'monitoring_batches': 5,
                   'nvis': 850,
                   'nhid': 1700,
                   'max_epochs': 10,
                   'save_path': '.'}
layer1_yaml = layer1_yaml % (hyper_params_l1)
print layer1_yaml
#
# train = yaml_parse.load(layer1_yaml)
# train.main_loop()
#
#
# mlp_yaml = open('dl1-copy.yaml', 'r').read()
# hyper_params_mlp = {'batch_size': 20,
# 'max_epochs': 10,
# 'nvis': 850,
#                     'n_classes': 2,
#                     'save_path': '.'}
# mlp_yaml = mlp_yaml % (hyper_params_mlp)
# # print mlp_yaml
# train = yaml_parse.load(mlp_yaml)
# train.main_loop()
