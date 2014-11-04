__author__ = 'Jackal'

#from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse

@no_debug_mode
def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()


def train(yaml_file_path, save_path):
    yaml = open("{0}/dl2_l2_rbm.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'nvis' : 1700,
                    'detector_layer_dim': 400,
                    'monitoring_batches': 10,
                    'max_epochs': 300,
                    'batch_size' : 30,
                    'save_path': save_path}

    yaml = yaml % (hyper_params)
    # print yaml
    train_yaml(yaml)


def train_dbm():
    # skip.skip_if_no_data()
    # yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dbm_demo'))
    # save_path = os.path.dirname(os.path.realpath(__file__))

    train(".", ".")


if __name__ == '__main__':
    train_dbm()