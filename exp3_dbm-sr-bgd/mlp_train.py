__author__ = 'Jackal'

from pylearn2.config import yaml_parse

MAX_EPOCHS = 2000


def train_mlp2():
    path = "mlp_tutorial_part_2.yaml"

    with open(path, 'r') as f:
        train_2 = f.read()
    hyper_params = {'dim_h0': 1500,
                    'dim_h1': 1500,
                    'sparse_init_h1': 15,
                    'max_epochs': MAX_EPOCHS,
                    'save_path': '.'}
    train_2 = train_2 % (hyper_params)

    train_2 = yaml_parse.load(train_2)
    train_2.main_loop()


def train_mlp3(data_path,
               dim_v=850,
               dim_h=1700):
    save_path = "mlp3-{}-{}-on-" + data_path
    save_path = save_path.format(dim_h, dim_h)
    dim_h1 = dim_h0 = dim_h
    #save_path.format(dim_h0, dim_h1)

    path = "mlp_tutorial_part_3.yaml"
    with open(path, 'r') as f:
        train_2 = f.read()

    hyper_params = {'data_path': data_path,
                    'nvis': dim_v,
                    'dim_h0': dim_h0,
                    'dim_h1': dim_h1,
                    #'sparse_init_h1': 15,
                    'max_epochs': MAX_EPOCHS,
                    'save_path': save_path}
    train_2 = train_2 % (hyper_params)
    # print train_2

    train_2 = yaml_parse.load(train_2)
    print "save to {}".format(save_path)
    #train_2.main_loop()
    return save_path


def train_mlp4(data_path,
               dim_v=850,
               dim_h=1700):

    save_path = "mlp4-{}-{}-on-".format(dim_h, dim_h) + datapath
    dim_h1 = dim_h0 = dim_h
    save_path.format(dim_h0, dim_h1)

    path = "mlp_tutorial_part_4.yaml"
    with open(path, 'r') as f:
        train_2 = f.read()

    hyper_params = {'data_path': data_path,
                    'nvis': dim_v,
                    'dim_h0': dim_h0,
                    'dim_h1': dim_h1,
                    # 'sparse_init_h1': 15,
                    'max_epochs': MAX_EPOCHS,
                    'save_path': save_path}
    train_2 = train_2 % (hyper_params)

    train_2 = yaml_parse.load(train_2)
    print "save to {}".format(save_path)
    train_2.main_loop()
    return save_path

def train_mymlp0(
        data_path,
        save_path='./mlp-pretrain0.pkl'
):
    path = "mlp-pretrain0.yaml"

    with open(path, 'r') as f:
        train_2 = f.read()

    hyper_params = {'dim_h2': 1500,
                    'data_path': data_path,
                    'sparse_init_h1': 15,
                    'max_epochs': MAX_EPOCHS,
                    'save_path': save_path}
    train_2 = train_2 % (hyper_params)

    train_2 = yaml_parse.load(train_2)

    train_2.main_loop()


def cross_valid(times, dim_h):
    datapath = "feature850-2-{}.pkl"
    #savepath = "./mlp4_{}.{}-on-{}"
    result = []
    for i in range(times):
        data_path = datapath.format(str(i+1))
        #save_path = savepath.format(dim_h, dim_h, data_path)
        print data_path
        #result.append(save_path)
        result.append(train_mlp3(data_path=data_path, dim_h=dim_h))
    return result


datapath = "feature850-2-{}.pkl".format(str(1))
dim_h = 1700
cross_n = 9
# savepath = "./mlp4_1700.1700-on-{}".format(datapath)
# savepath = "./mlp-pretain0.pkl"
results = []
# results.append(train_mlp3(datapath))
# results.append(train_mlp4(datapath))
results += cross_valid(cross_n, dim_h)

#train_mymlp0(datapath)
#train_mlp4(datapath, savepath)
# resultfiles = cross_valid(9)
# print resultfiles





