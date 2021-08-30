"""
Diogo Amorim, 2018-07-10
Evaluate Predictions - Vnet
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_dataset(path, h5=True):
    #print("Loading dataset... Shape:")
    file = h5py.File(path, 'r')
    if h5:
        data = file.get('data')
        truth = file.get('truth')
    else:
        data = file.get('data').value
        truth = file.get('truth').value
    # print(data.shape)
    return data, truth

def load_predictions(path):
    path = path + 'predictions_vnet.h5'
    print("Loading predictions... Shape:")
    data = h5py.File(path, 'r')
    predictions = data.get('predictions').value
    print(predictions.shape)
    return predictions


def print_prediction(test, pred, m, slice):
    fig = plt.figure()
    y = fig.add_subplot(1, 2, 1)
    y.imshow(test[m, :, :, slice], cmap='gray')
    y = fig.add_subplot(1, 2, 2)
    y.imshow(pred[m, :, :, slice], cmap='gray')
    plt.show()

# Modifying the directory variables to run at the AWS Ubuntu server

# Structure of the working directory 
# ./Liver
# ./Liver/train
# ./Liver/val
# ./Liver/test
# ./Liver/datasets

data_dir = "/home/ubuntu/Liver/"
save_dir = data_dir + "datasets/"
test_dir = save_dir + "test_data.h5"

predictions = load_predictions(save_dir)
x_test, y_test = load_dataset(test_dir)
x_test = np.squeeze(np.array(np.split(x_test, 8, axis=0)))
y_test = np.squeeze(np.array(np.split(y_test, 8, axis=0)))

print_prediction(y_test, predictions, 5, 50)
