__author__ = 'Jackal'

from pylearn2.models.mlp import MLP, CompositeLayer
from pylearn2.space import CompositeSpace
from theano.compat.python2x import OrderedDict
from theano import shared
import numpy as np
import theano.tensor as T


class MLPWithSource(MLP):
    def __init__(self, *args, **kwargs):
        input_src = kwargs.get('input_source', 'features')
        target_src = kwargs.get('target_source', 'targets')
        super(MLPWithSource, self).__init__(*args, **kwargs)
        # self.input_source = input_src
        self.target_source = target_src
        print self.get_input_source(), self.get_target_source()


    def get_input_source(self):
        return self.input_source

    def get_target_source(self):
        return self.target_source

    def get_weights(self):
        return self.layers[-1].get_weights()

    def get_weights_format(self):
        return self.layers[-1].get_weights_format()

    def get_biases(self):
        return self.layers[-1].get_biases()

    def get_final_output(self, dataset):
        res = []
        for i, features in enumerate(dataset):
            tres = shared(features)
            for layer in self.layers[0:-1]:
                layer = layer.layers[i]
                tres = layer.fprop(tres)
            res.append(tres)
        return T.concatenate(res, axis=1)

    # def get_final_output(self, dataset):
    #     res = []
    #     for i, features in enumerate(dataset):
    #         tres = shared(features)
    #         tres = shared((self.layers[0].layers[i].fprop(tres)).eval())
    #         tres = shared((self.layers[1].layers[i].fprop(tres)).eval())
    #
    #         # for layer in self.layers[0:-1]:
    #         #     layer = layer.layers[i]
    #         #     tres = layer.fprop(tres)
    #
    #         res.append(tres)
    #     return T.concatenate(res, axis=1)

class CompositeLayerWithSource(CompositeLayer):
    def get_input_source(self):
        return tuple([layer.get_input_source() for layer in self.layers])

    def get_target_source(self):
        return tuple([layer.get_target_source() for layer in self.layers])

    def set_input_space(self, space):
        self.input_space = space

        for layer, component in zip(self.layers, space.components):
            layer.set_input_space(component)

        self.output_space = CompositeSpace(tuple(layer.get_output_space()
                                                 for layer in self.layers))

    def fprop(self, state_below):
        return tuple(layer.fprop(component_state) for
                     layer, component_state in zip(self.layers, state_below))

    def get_monitoring_channels(self):
        return OrderedDict()

