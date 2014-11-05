__author__ = 'Jackal'

from pylearn2.config import yaml_parse


def train_mlp2():
    path = "mlp_tutorial_part_2.yaml"

    with open(path, 'r') as f:
        train_2 = f.read()
    hyper_params = {'dim_h0': 1500,
                    'dim_h1': 1500,
                    'sparse_init_h1': 15,
                    'max_epochs': 5000,
                    'save_path': '.'}
    train_2 = train_2 % (hyper_params)

    train_2 = yaml_parse.load(train_2)
    train_2.main_loop()

def train_mlp3():
    path = "mlp_tutorial_part_3.yaml"

    with open(path, 'r') as f:
        train_2 = f.read()
    hyper_params = {'dim_h0': 2500,
                    'dim_h1': 2500,
                    'sparse_init_h1': 15,
                    'max_epochs': 5000,
                    'save_path': './mlp3_2500.2500-1406-best.pkl'}
    train_2 = train_2 % (hyper_params)

    train_2 = yaml_parse.load(train_2)
    train_2.main_loop()


def train_mlp4():
    path = "mlp_tutorial_part_4.yaml"

    with open(path, 'r') as f:
        train_2 = f.read()

    hyper_params = {'dim_h0': 1500,
                    'dim_h1': 1500,
                    'sparse_init_h1': 15,
                    'max_epochs': 5000,
                    'save_path': '.'}
    train_2 = train_2 % (hyper_params)

    train_2 = yaml_parse.load(train_2)
    train_2.main_loop()

train_mlp3()


