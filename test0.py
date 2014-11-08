__author__ = 'Jackal'

#from pylearn2.datasets.mnist import MNIST
from pylearn2.config import yaml_parse


d = {'h':'hello', 'w':'world!'}
print d

c = d.pop(1, 'default')
print c