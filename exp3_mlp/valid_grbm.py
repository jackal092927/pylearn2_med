__author__ = 'Jackal'

from datasets.cin_feature2_split import CIN_FEATURE2


MAX_EPOCHS_UNSUPERVISED = 1
MAX_EPOCHS_SUPERVISED = 2

from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.mlp import Default
from pylearn2.models.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import GaussianBinaryRBM, RBM
from pylearn2.models.softmax_regression import SoftmaxRegression
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.costs.ebm_estimation import SMD, CDk
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.train import Train


def get_dataset_cin():
    """
    The toy dataset is only meant to used for testing pipelines.
    Do not try to visualize weights on it. It is not picture and
    has no color channel info to support visualization
    """
    trainset = CIN_FEATURE2('train')
    testset = CIN_FEATURE2('test')

    return trainset, testset


def get_autoencoder(structure):
    n_input, n_output = structure
    config = {
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True,
        'act_enc': 'sigmoid',
        'act_dec': 'sigmoid',
        'irange': 0.001,
    }
    return Autoencoder(**config)


def get_denoising_autoencoder(structure):
    n_input, n_output = structure
    curruptor = BinomialCorruptor(corruption_level=0.5)
    config = {
        'corruptor': curruptor,
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True,
        'act_enc': 'sigmoid',
        'act_dec': 'sigmoid',
        'irange': 0.001,
    }
    return DenoisingAutoencoder(**config)


def get_grbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange": 0.05,
        "energy_function_class": GRBM_Type_1,
        "learn_sigma": True,
        "init_sigma": .4,
        "init_bias_hid": -2.,
        "mean_vis": False,
        "sigma_lr_scale": 1e-3
    }

    return GaussianBinaryRBM(**config)


def get_rbm(structure):
    n_input, n_output = structure
    config = {
        'nvis': n_input,
        'nhid': n_output,
        "irange": 0.05,
        "init_bias_hid": -2.,
    }

    return RBM(**config)


def get_logistic_regressor(structure):
    n_input, n_output = structure

    layer = SoftmaxRegression(n_classes=n_output, irange=0.02, nvis=n_input)

    return layer


def get_layer_trainer_logistic(layer, trainset):
    # configs on sgd

    config = {'learning_rate': 0.1,
              'cost': Default(),
              'batch_size': 10,
              'monitoring_batches': 10,
              'monitoring_dataset': trainset,
              'termination_criterion': EpochCounter(max_epochs=MAX_EPOCHS_SUPERVISED),
              'update_callbacks': None
    }

    train_algo = SGD(**config)
    model = layer
    return Train(model=model,
                 dataset=trainset,
                 algorithm=train_algo,
                 extensions=None)


def get_layer_trainer_sgd_autoencoder(layer, trainset):
    # configs on sgd
    train_algo = SGD(
        learning_rate=0.1,
        cost=MeanSquaredReconstructionError(),
        batch_size=10,
        monitoring_batches=10,
        monitoring_dataset=trainset,
        termination_criterion=EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
        update_callbacks=None
    )

    model = layer
    extensions = None
    return Train(model=model,
                 algorithm=train_algo,
                 extensions=extensions,
                 dataset=trainset,
                 save_path='my_train.pkl')


def get_layer_trainer_sgd_grbm(layer, trainset, testset):
    train_algo = SGD(
        learning_rate=1e-1,
        batch_size=10,
        # "batches_per_iter" : 2000,
        monitoring_batches=20,
        monitoring_dataset={'train': trainset, 'test': testset},
        cost=SMD(corruptor=GaussianCorruptor(stdev=0.8)),
        termination_criterion=MonitorBased(
            prop_decrease=.001, N=10,
            channel_name='test_reconstruction_error'
        )
        #EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
    )
    model = layer
    extensions = [MonitorBasedLRAdjuster(dataset_name='test')]
    return Train(model=model,
                 dataset=trainset,
                 algorithm=train_algo,
                 extensions=extensions,
                 save_path='valid_grbm.pkl',
                 save_freq=1
                 )


def get_layer_trainer_sgd_rbm(layer, trainset):
    train_algo = SGD(
        learning_rate=1e-1,
        batch_size=5,
        # "batches_per_iter" : 2000,
        monitoring_batches=20,
        monitoring_dataset=trainset,
        cost=CDk(nsteps=10),
        termination_criterion=EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
    )
    model = layer
    extensions = [MonitorBasedLRAdjuster(channel_name='reconstruction_error')]
    return Train(model=model, algorithm=train_algo,
                 save_path='l2_rbm.pkl', save_freq=1,
                 extensions=extensions, dataset=trainset)


def main(args=None):
    """
    args is the list of arguments that will be passed to the option parser.
    The default (None) means use sys.argv[1:].
    """
    # parser = OptionParser()
    # parser.add_option("-d", "--data", dest="dataset", default="toy",
    # help="specify the dataset, either cifar10, mnist or toy")
    # (options, args) = parser.parse_args(args=args)

    # if options.dataset == 'toy':
    #     trainset, testset = get_dataset_toy()
    #     n_output = 2
    # elif options.dataset == 'cifar10':
    #     trainset, testset, = get_dataset_cifar10()
    #     n_output = 10
    #
    # elif options.dataset == 'mnist':
    #     trainset, testset, = get_dataset_mnist()
    #     n_output = 10
    #
    # elif options.dataset == 'cin':
    #     trainset, testset, = get_dataset_cin()
    #     n_output = 2
    #
    # else:
    #     NotImplementedError()

    trainset, testset, = get_dataset_cin()
    n_output = 2

    design_matrix = trainset.get_design_matrix()
    n_input = design_matrix.shape[1]

    # build layers
    layers = []
    structure = [[n_input, 3000], [1700, 200], [3000, n_output]]
    # layer 0: gaussianRBM
    layers.append(get_grbm(structure[0]))
    # layer 1: denoising AE
    # layers.append(get_rbm(structure[1]))
    # layer 2: AE
    #layers.append(get_rbm(structure[2]))
    # layer 3: logistic regression used in supervised training
    layers.append(get_logistic_regressor(structure[2]))


    # construct training sets for different layers
    trainset = [trainset,
                TransformerDataset(raw=trainset, transformer=layers[0]),
                #TransformerDataset(raw=trainset, transformer=StackedBlocks(layers[0:2]))
                # , TransformerDataset(raw=trainset, transformer=StackedBlocks(layers[0:3]))
    ]

    # construct layer trainers
    layer_trainers = []
    layer_trainers.append(get_layer_trainer_sgd_grbm(layers[0], trainset[0], testset))
    #layer_trainers.append(get_layer_trainer_sgd_rbm(layers[1], trainset[1]))
    #layer_trainers.append(get_layer_trainer_sgd_autoencoder(layers[2], trainset[2]))
    layer_trainers.append(get_layer_trainer_logistic(layers[1], trainset[1]))

    # unsupervised pretraining
    for i, layer_trainer in enumerate(layer_trainers[0:2]):
        print '-----------------------------------'
        print ' Unsupervised training layer %d, %s' % (i, layers[i].__class__)
        print '-----------------------------------'
        layer_trainer.main_loop()

    print '\n'
    print '------------------------------------------------------'
    print ' Unsupervised training done! Start supervised training...'
    print '------------------------------------------------------'
    print '\n'

    # supervised training -- finetunning?
    # layer_trainers[-1].main_loop()


if __name__ == '__main__':
    main()
