__author__ = 'Jackal'
import sys
from pylearn2.config import yaml_parse

MAX_EPOCHS = 2000


def train_mlp2(data_path,
               dim_v=1406,
               dim_h=1700):
    save_path = "mlp2-{}-{}-on-" + data_path
    save_path = save_path.format(dim_h, dim_h)
    dim_h1 = dim_h0 = dim_h
    # save_path.format(dim_h0, dim_h1)

    path = "mlp_tutorial_part_2.yaml"
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
    train_2.main_loop()
    return save_path


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
    train_2.main_loop()
    return save_path


def train_mlp4(data_path,
               dim_v=850,
               dim_h=1700):

    save_path = "mlp4-{}-{}-on-".format(dim_h, dim_h) + data_path
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

dir_path = "mlp-1700-wd.0005-on-feature850-2-fold"
save_path_tmp = "../results/"+dir_path+"/mlp{}{}{}-wd{}-on-{}"
def mlpwd_train_850(filename,
                    dim_v=850,
                    dim_h=1700,
                    wd=.0005,
                    foldi=1):
    save_path = save_path_tmp.format("-" + str(dim_h), "", "", wd, filename)
    dim_h1 = dim_h0 = dim_h

    yaml_path = "mlp_tutorial_part_4.yaml"
    with open(yaml_path, 'r') as f:
        train_2 = f.read()

    hyper_params = {'foldi': foldi,
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
        #print data_path
        #result.append(save_path)
        result.append(train_mlp4(data_path=data_path, dim_h=dim_h))
    return result


def main():
    # datapath = "feature850-2-{}.pkl".format(str(1))
    # dim_h = 1700
    # cross_n = 9
    # results = []
    # results += cross_valid(cross_n, dim_h)
    # data_path = "feature1406-2-1.pkl"

    filename_str = "feature850-2-fold{}.pkl"
    foldi = 1
    if len(sys.argv) >= 2:
        foldi = int(sys.argv[1])
    filename = filename_str.format(str(foldi))
    mlpwd_train_850(filename=filename, foldi=foldi)


if __name__ == '__main__':
    main()
    print "#####DONE#####"



