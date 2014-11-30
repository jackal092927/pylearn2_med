from cin_feature2 import CIN_FEATURE2

__author__ = 'Jackal'

from pylearn2.corruption import BinomialCorruptor
from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.mlp import Default
from pylearn2.models.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import GaussianBinaryRBM, RBM
from pylearn2.models.softmax_regression import SoftmaxRegression
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.termination_criteria import EpochCounter
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
from pylearn2.blocks import StackedBlocks
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


def get_logistic_regressor(structure):
    n_input, n_output = structure

    layer = SoftmaxRegression(n_classes=n_output, irange=0.02, nvis=n_input)

    return layer

def main():
    pass










if __name__ == '__main__':
    main()
