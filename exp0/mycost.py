__author__ = 'Jackal'
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
import theano.tensor as T

class MyCostSubclass (Cost, DefaultDataSpecsMixin):

    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.get_outputs(inputs)
        loss = -(targets * T.log(outputs)).sum(axis=1)  # use the NLL loss function
        loss = loss.mean()

        return loss
