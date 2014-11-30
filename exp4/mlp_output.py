import sys

__author__ = 'Jackal'

from datasets.cin_feature2_composite import *
from pylearn2.datasets.transformer_dataset import TransformerDataset
from theano import shared
from pylearn2.utils import serial
from pylearn2.gui import get_weights_report
import numpy as np
import gc


def saveaspkl(data, path):
    with open(path, 'wb') as f:
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)

dir_path_dict = {'850': "../results/mlp-1700-wd.0005-on-feature850-2-fold/",
                 '1406': "",
                 '2086': "../results/mlpws-1700-1200-700-wd0.0005-on-feature2086-2/"}

path_str_dict = {'850' : [dir_path_dict['850'] + "mlp-1700-wd0.0005-on-feature850-2-fold{}.pkl",
                          dir_path_dict['850'] + "feature1700-850-2-fold{}_output.pkl"],
                 '1406': [dir_path_dict['1406'] + "mlp-1700-1200-wd0.0005-on-feature1406-2-fold{}.pkl",
                          dir_path_dict['1406'] + "feature2900-1406-2-fold{}_output.pkl"],
                 '2086': [dir_path_dict['2086'] + "mlpws-1700-1200-700-wd0.0005-on-feature2086-2-{}.pkl",
                          dir_path_dict['2086'] + "feature3600-2086-2-fold{}_output.pkl"]}
class MLP_output():

    # model_path_str =  dir_path + "mlp-1700-1200-wd0.0005-on-feature1406-2-fold{}.pkl"
    # output_path_str = dir_path + "feature2900-1406-2-fold{}_output.pkl"


    def __init__(self, foldi, featuren=1406, with_original=True):
        self.model_path_str, self.output_path_str = path_str_dict[str(featuren)]
        self.foldi = foldi
        self.featuren = featuren
        self.get_dataset_cin()
        self.model_path = self.model_path_str.format(foldi)
        self.output_path = self.output_path_str.format(foldi)
        self.with_original = with_original

    def get_dataset_cin(self):
        """
        The toy dataset is only meant to used for testing pipelines.
        Do not try to visualize weights on it. It is not picture and
        has no color channel info to support visualization
        """

        # if self.featuren == 1406:
        #     trainset = CIN_FEATURE1406_2('train', self.foldi).get_data()
        #     validset = CIN_FEATURE1406_2('valid', self.foldi).get_data()
        #     testset = CIN_FEATURE1406_2('test', self.foldi).get_data()
        # if self.featuren == 2086:
        #     trainset = CIN_FEATURE2086_2('train', self.foldi).get_data()
        #     validset = CIN_FEATURE2086_2('valid', self.foldi).get_data()
        #     testset = CIN_FEATURE2086_2('test', self.foldi).get_data()

        trainset = get_CIN_FEATURE(featuren=self.featuren, which_set='train', foldi=self.foldi).get_data()
        validset = get_CIN_FEATURE(featuren=self.featuren, which_set='valid', foldi=self.foldi).get_data()
        testset = get_CIN_FEATURE(featuren=self.featuren, which_set='test', foldi=self.foldi).get_data()

        self.trainX, self.trainy = trainset[0:-1], trainset[-1]
        self.validX, self.validy = validset[0:-1], validset[-1]
        self.testX, self.testy = testset[0:-1], testset[-1]
        self.trainX = [np.vstack(dataset) for dataset in zip(self.trainX, self.validX)]
        self.trainy = np.vstack([self.trainy, self.validy])


    def mlp_fprop(self, dataset, model):
        feature850 = shared(dataset.feature850, name='state0')
        feature556 = shared(dataset.feature556, name='state1')
        layer00 = (model.layers[0]).layers[0]
        layer10 = (model.layers[1]).layers[0]
        output = layer10.fprop(layer00.fprop(feature850))
        output0 = output.eval()

        layer01 = (model.layers[0]).layers[1]
        layer11 = (model.layers[1]).layers[1]
        output = layer11.fprop(layer01.fprop(feature556))
        output1 = output.eval()

        return np.hstack((output0, output1))


    def get_model_output(self, model_path):
        model = serial.load(model_path)
        train_set = self.trainX
        test_set = self.testX

        # X, y = test_set.X, teswt_set.y
        # ts = TransformerDataset(raw=train_set, transformer=model)

        train_output = (model.get_final_output(train_set)).eval()
        test_output = (model.get_final_output(test_set)).eval()
        if self.with_original:
            train_output = np.hstack([train_output, np.hstack(train_set)])
            test_output = np.hstack([test_output, np.hstack(test_set)])
            print train_output.shape
            print test_output.shape
        # train_output = mlp_fprop(train_set, model)
        # test_output = mlp_fprop(test_set, model)
        train_y = self.trainy
        test_y = self.testy

        # output = ts.get_batch_design(batch_size=1140)

        return train_output, test_output, train_y, test_y


    def output_feature(self):
        train, test, train_y, test_y = self.get_model_output(model_path=self.model_path)
        data = ((train, train_y), (test, test_y))
        # path = "mlpws-1700-1200-wd0.0005-on-feature1406-2-1-shuffle_output.pkl"
        print "save to:\t", self.output_path
        saveaspkl(data, self.output_path)

    # def loop_output_features(self, file_str=None, model_str=None):
    #     path_tmp = "../results/mlpws-1700-1200-wd0.0005-on-feature1406-2-shuffle/mlpws-1700-1200-wd0.0005-on-{}"
    #     model_tmp = path_tmp.format("feature1406-2-{}-shuffle.pkl")
    #     out_tmp = "feature1406-2-{}-shuffle_output.pkl"
    #     for i in range(1, 9):
    #         s = str(i + 1)
    #         output_path = out_tmp.format(s)
    #         model_path = model_tmp.format(s)
    #         print model_path, "\t-->\t"
    #         self.output_feature(model_path=model_path, output_path=output_path)
    #         print output_path

def main():
    foldi = 1
    featuren = 850
    if len(sys.argv) >= 2:
        foldi = sys.argv[1]
    for i in range(10):
        MLP_output(foldi=i+1, featuren=featuren).output_feature()

if __name__ == '__main__':
    main()
