__author__ = 'Jackal'

from pylearn2.config import yaml_parse
def pre_train():
    layer1_yaml = open('dl2_l1_grbm-sgd.yaml', 'r').read()
    hyper_params_l1 = {'batch_size': 30,
                       'monitoring_batches': 30,
                       'nvis': 850,
                       'nhid': 1700,
                       'save_path': '.'}
    layer1_yaml = layer1_yaml % (hyper_params_l1)
    print layer1_yaml

    train = yaml_parse.load(layer1_yaml)
    train.main_loop()

pre_train()