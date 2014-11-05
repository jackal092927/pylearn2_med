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


def train_mlp3(data_path,
               save_path='./mlp3_2500.2500-1406-best.pkl'):
    path = "mlp_tutorial_part_3.yaml"

    with open(path, 'r') as f:
        train_2 = f.read()
    hyper_params = {'data_path': data_path,
                    'dim_h0': 2500,
                    'dim_h1': 2500,
                    'sparse_init_h1': 15,
                    'max_epochs': 1,
                    'save_path': save_path}
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


def cross_valid(times):
    datapath = "feature1406-2-{}.pkl"
    savepath = "./mlp3_2500.2500-on-{}"
    result = []
    for i in range(times):
        data_path = datapath.format(str(i+1))
        save_path = savepath.format(data_path)
        print data_path, save_path
        result.append(save_path)
        train_mlp3(data_path, save_path)
    return result


# train_mlp3()
resultfiles = cross_valid(9)
print resultfiles





