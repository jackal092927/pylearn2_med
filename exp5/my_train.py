from pylearn2.scripts.gsn_example import MAX_EPOCHS

__author__ = 'Jackal'

from pylearn2.config import yaml_parse

MAX_EPOCHS = 1000

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
    # savepath = "./mlp4_{}.{}-on-{}"
    result = []
    for i in range(times):
        data_path = datapath.format(str(i + 1))
        #save_path = savepath.format(dim_h, dim_h, data_path)
        #print data_path
        #result.append(save_path)
        # result.append(train_mlp4(data_path=data_path, dim_h=dim_h))
    return result

def svm_train():
    yaml_path = "svm.yaml"
    with open(yaml_path, 'r') as f:
        model = f.read()
    print model
    train = yaml_parse.load(model)
    train.main_loop()



if __name__ == '__main__':
    # datapath = "feature850-2-{}.pkl".format(str(1))
    # dim_h = 1700
    # cross_n = 9
    # results = []
    # results += cross_valid(cross_n, dim_h)

    data_path = "feature1406-2-1.pkl"
    svm_train()