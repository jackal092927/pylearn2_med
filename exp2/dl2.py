__author__ = 'Jackal'

from pylearn2.config import yaml_parse

# layer1_yaml = open('dae_l1.yaml', 'r').read()
# hyper_params_l1 = {'train_stop': 50000,
#                    'batch_size': 100,
#                    'monitoring_batches': 5,
#                    'nhid': 500,
#                    'max_epochs': 10,
#                    'save_path': '.'}
# layer1_yaml = layer1_yaml % (hyper_params_l1)
# print layer1_yaml
#
# # train = yaml_parse.load(layer1_yaml)
# # train.main_loop()
#
# layer2_yaml = open('dae_l2.yaml', 'r').read()
# hyper_params_l2 = {'train_stop': 50000,
#                    'batch_size': 100,
#                    'monitoring_batches': 5,
#                    'nvis': hyper_params_l1['nhid'],
#                    'nhid': 500,
#                    'max_epochs': 10,
#                    'save_path': '.'}
# layer2_yaml = layer2_yaml % (hyper_params_l2)
# print layer2_yaml
#
# # train = yaml_parse.



# layer1_yaml = open('l1_grbm.yaml', 'r').read()
# hyper_params_l1 = {'batch_size': 30,
#                    'monitoring_batches': 30,
#                    'nvis': 850,
#                    'nhid': 1700,
#                    'save_path': '.'}
# layer1_yaml = layer1_yaml % (hyper_params_l1)
# print layer1_yaml
#
# train = yaml_parse.load(layer1_yaml)
# train.main_loop()


mlp_yaml = open('dl2.yaml', 'r').read()
hyper_params_mlp = {'batch_size': 1140,
                    'bgd_batch_size' : 1140,
                    'max_epochs': 5,
                    'nvis': 850,
                    'n_classes': 2,
                    'save_path': '.'}
mlp_yaml = mlp_yaml % (hyper_params_mlp)
# print mlp_yaml
train = yaml_parse.load(mlp_yaml)
train.main_loop()

