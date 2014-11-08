#!/usr/bin/env python
__author__ = 'Jackal'

import sys
from pylearn2.utils import serial
from pylearn2.gui import get_weights_report
import numpy as np
import gc

def my_monitor(models=None, n=10, model_str="mlp3-1700-1700-on-feature850-2-{}.pkl"):
    if models is None:
        models = []
        for i in range(1,n):
            models.append(model_str.format(str(i)))

    errors = []
    for model_path in models:
        # if len(models) > 1:
        print model_path
        model = serial.load(model_path)
        monitor = model.monitor
        del model
        gc.collect()
        channels = monitor.channels
        keys = ["test_y_misclass"]#, "train_y_misclass", "valid_y_misclass"]
        for key in keys:
            value = channels[key].val_record[-1]
            print key, ':', value
            errors.append(value)


    return errors

    if not hasattr(monitor, '_epochs_seen'):
        print 'old file, not all fields parsed correctly'
    else:
        print 'epochs seen: ', monitor._epochs_seen
    print 'time trained: ', max(channels[key].time_record[-1] for key in
                                channels)
    for key in sorted(channels.keys()):
        print key, ':', channels[key].val_record[-1]

def my_show_weights(model_path, rescale='individual', border=False, out=None):
    pv = get_weights_report.get_weights_report(model_path=model_path,
                                               rescale=rescale,
                                               border=border)

    if out is None:
        pv.show()
    else:
        pv.save(out)


if __name__ == '__main__':
    # errors = my_monitor(
    #     models=None,
    #     model_str=sys.argv[1],
    #     n=int(sys.argv[2])
    # )
    # print errors
    # print np.mean(errors)
    # print np.std(errors)

    model_path = "mlpws-1700-1200-on-feature1406-2-1.pkl"
    model_str = "mlpws-1700-1200-on-feature1406-2-{}.pkl"
    errors = my_monitor(model_str=model_str, n=9)
    print errors
    print np.mean(errors)
    print np.std(errors)

    # my_show_weights(model_path=model_path)



