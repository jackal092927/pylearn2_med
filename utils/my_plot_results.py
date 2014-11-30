import numpy as np

__author__ = 'Jackal'

import matplotlib.pyplot as plt

class Channels():
    def __init__(self):
        self.channel = {
            'test_y_misclass': [],
            'train_y_misclass': [],
            'valid_y_misclass': [],
            'valid_objective': []
        }
        # self.test_y_misclass = []
        # self.train_y_misclass = []
        # self.valid_y_misclass = []
        # self.test_y_misclass = []
        # self.test_y_misclass = []




def load(path):
    file_channels = []
    current = Channels()
    with open(path) as f:
        for i, line in enumerate(f):
            if len(line) <= 1:
                continue
            s = line.strip()
            if line.startswith("save to mlpws-1700-1200"):
                file_channels.append(current)
                current = Channels()
                continue
                #return ys
            strs = s.split(':')
            if len(strs) < 2:
                # print line
                continue
            key, value = strs[0], strs[1]
            if key in current.channel:
                current.channel[key].append(value)
            # if key == 'test_y_misclass:':
            #     current.channel[key].append(value)
            # elif key == 'train_y_misclass:':
            #     current.test_y_misclass.append(value)
            # elif key == 'train'
    file_channels.append(current)

    return file_channels




def getkv(str):
    str = str.split()
    key, value = str.split(':')



if __name__ == '__main__':
    # path = '/Users/Jackal/Work/pylearn/pylearn2/pylearn2/scripts/med_ml/results/mlp-composite-on-feature1406-2-[2-5].txt'
    path = '/Users/Jackal/Work/pylearn/pylearn2/pylearn2/scripts/med_ml/results/mlp-composite-on-feature1406-2-[5-9].txt'
    file_channels = load(path)
    for result in file_channels:
        y1 = result.channel['train_y_misclass']
        y2 = result.channel['valid_y_misclass']
        y3 = result.channel['test_y_misclass']
        #y4 = result.channel['valid_objective']

        n = min(len(y1), len(y2), len(y3))
        if n < 1:
            continue
        y1 = y1[0:n]
        y2 = y2[0:n]
        y3 = y3[0:n]
        x = xrange(n)
        plt.plot(x, y1, 'bs--', x, y2, 'g^--', x, y3, 'r.--')
        plt.show()


    # x = xrange(0, len(ys))
    # plt.plot(x, ys, 'r--')
    # plt.show()
