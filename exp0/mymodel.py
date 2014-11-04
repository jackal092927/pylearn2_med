__author__ = 'Jackal'

from pylearn2.models.model import Model
import numpy as np
import theano.tensor as T
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX

class MyModelClass(Model):
    def __init__(self, nvis, nclasses):
        super(MyModelClass, self).__init__()


        self.nvis = nvis
        self.nclasses = nclasses

        W_value = np.random.uniform(size=(self.nvis, self.nclasses))
        self.W = sharedX(W_value, 'W')
        b_value = np.zeros(self.nclasses)
        self.b = sharedX(b_value, 'b')
        # Some parameter initialization using *args and **kwargs
        # ...
        self._params = [ self.W, self.b ]

        self.input_space = VectorSpace(dim=self.nvis)
        # This one is necessary only for supervised learning
        self.output_space = VectorSpace(dim=self.nclasses)




    def get_outputs(self, inputs):
        return T.nnet.softmax(T.dot(inputs, self.W) + self.b)

