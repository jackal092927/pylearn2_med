__author__ = 'Jackal'

from utils.loader import *

def svm_cv():
    dir_key, data_key = ('850', '1700')
    # dir_path = dir_path_dict[dir_key]
    # data_str = data_str_dict[data]

    data_str = dir_path_dict[dir_key] + data_str_dict[data_key]
    output_path = "SVM-on-feature-{}-2-fold.txt".format(dir_key)

    errors = []
    sensis = []
    specis = []
    for i in range(10):
        data_path = data_str.format(i + 1)
        print data_path
        trainset, testset = get_dataset(data_path=data_path)
