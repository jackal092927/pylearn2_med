__author__ = 'Jackal'

import sys
sys.path.append('..')
from pylearn2.config import yaml_parse
from pylearn2.space import VectorSpace
from datasets.cin_feature2_composite import CIN_FEATURE2
from mlp_with_source import MLPWithSource, CompositeLayerWithSource
from pylearn2.costs.mlp import Default
from pylearn2.models.mlp import Linear
from pylearn2.space import CompositeSpace
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.bgd import BGD



MAX_EPOCHS = 2000

def train_mlp_with_source(data_path,
               dim_h=[1700, 1200],
               wd_coeff=.0005):
    save_path = "mlpws-{}-{}-wd{}-on-" + data_path
    save_path = save_path.format(dim_h[0], dim_h[1], wd_coeff)
    dim_850, dim_556 = dim_h
    # save_path.format(dim_h0, dim_h1)

    path = "mlp-composite0.yaml"
    with open(path, 'r') as f:
        train_2 = f.read()

    hyper_params = {'data_path': data_path,
                    # 'nvis': dim_v,
                    'dim_h0_850': dim_850,
                    'dim_h0_556': dim_556,
                    # 'sparse_init_h1': 15,
                    'wd_coeff': wd_coeff,
                    'max_epochs': MAX_EPOCHS,
                    'save_path': save_path}
    train_2 = train_2 % (hyper_params)
    # print train_2

    train_2 = yaml_parse.load(train_2)
    print "save to {}".format(save_path)
    train_2.main_loop()
    return save_path


def train_mlpws_1(data_path=None,
                          dim_h=[1700, 1200, 1400],
                          wd_coeff=.0005,
                          foldi=1):
    save_path = "mlpws-{}-{}-{}-wd{}-on-feature2086-2-{}.pkl"
    save_path = save_path.format(dim_h[0], dim_h[1], dim_h[2], wd_coeff, foldi)
    dim_850, dim_556, dim_680 = dim_h
    # save_path.format(dim_h0, dim_h1)

    path = "mlp-composite1.yaml"
    with open(path, 'r') as f:
        train_2 = f.read()

    hyper_params = {# 'data_path': data_path,
                    # 'nvis': dim_v,
                    'dim_h0_850': dim_850,
                    'dim_h0_556': dim_556,
                    'dim_h0_680': dim_680,
                    'foldi': foldi,
                    # 'sparse_init_h1': 15,
                    'wd_coeff': wd_coeff,
                    'max_epochs': MAX_EPOCHS,
                    'save_path': save_path}
    train_2 = train_2 % (hyper_params)
    # print train_2

    train_2 = yaml_parse.load(train_2)
    print "save to {}".format(save_path)
    train_2.main_loop()
    return save_path


def my_train():
    trainset = CIN_FEATURE2(which_set='train')
    validset = CIN_FEATURE2(which_set='valid')
    layers = []
    layers1 = []
    h1 = Linear(layer_name='h1', dim=850, irange=0.05)
    h2 = Linear(layer_name='h2', dim=556, irange=0.05)
    layers1.append(h1)
    layers1.append(h2)
    l1 = CompositeLayerWithSource(layer_name='c', layers=layers1)
    l2 = Linear(layer_name='o', dim=2, irange=0.05)
    layers.append(l1)
    layers.append(l2)

    input_space = CompositeSpace(components=[VectorSpace(dim=850), VectorSpace(dim=556)])
    input_source = ['feature850', 'feature556']
    model = MLPWithSource(batch_size=1140, layers=layers,
                          input_space=input_space, input_source=input_source)

    algorithm = BGD(conjugate=1,
                    # batch_size=1140,
                    line_search_mode='exhaustive',
                    cost=Default(),
                    termination_criterion=EpochCounter(max_epochs=MAX_EPOCHS))

    train = Train(dataset=trainset, model=model, algorithm=algorithm)
    train.main_loop()



def cross_valid(n_fold, dim_h, datapath):
    # datapath = "feature850-2-{}.pkl"
    # savepath = "./mlp4_{}.{}-on-{}"
    result = []
    for i in range(n_fold):
        data_path = datapath.format(str(i + 1))
        #save_path = savepath.format(dim_h, dim_h, data_path)
        #print data_path
        #result.append(save_path)
        result.append(train_mlp_with_source(data_path=data_path, dim_h=dim_h))
    return result


def main():
    # errors = my_monitor(
    # models=None,
    # model_str=sys.argv[1],
    #     n=int(sys.argv[2])
    # )

    # datapath = "feature850-2-{}.pkl".format(str(1))
    # dim_h = 1700
    n_fold = 9
    # results = []
    # results += cross_valid(cross_n, dim_h)

    # data_path = "feature1406-2-1.pkl"
    # train_mlp_with_source(data_path=data_path)

    # data_path = "feature1406-2-{}-shuffle.pkl"
    # dim_h = [1700, 1200]
    # cross_valid(n_fold=n_fold, dim_h=dim_h, datapath=data_path)
    train_mlpws_1(sys.argv[1])

if __name__ == '__main__':
    main()

