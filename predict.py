"""
Diogo Amorim, 2018-07-13
Predict Network Outcome - Vnet
"""

import numpy as np

from vnet import vnet
from utils import *


def predict(test, model):
    predictions = []
    m = test.shape[0]
    print('Starting predictions:')
    print("0/%i (0%%)" % m)

    for i in range(m):
        image = test[i][np.newaxis, :, :, :]
        prediction = model.predict(image, steps=1)
        prediction = np.squeeze(prediction)
        predictions.append(prediction)
        print("%i/%i (%i%%)" % (i + 1, m, ((i + 1) / m * 100)))

    predictions = np.array(predictions)

    print('Predictions obtained with shape:')
    print(predictions.shape)
    return predictions


def write_predictions(predicitons, path):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(name='predictions', data=predicitons)
    h5f.close()
    print('Predictions saved to ' + path)

# Modifying the directory variables to run at the AWS Ubuntu server

# Structure of the working directory 
# ./Liver
# ./Liver/train
# ./Liver/val
# ./Liver/test
# ./Liver/datasets

data_dir = "home/ubuntu/Liver/"
save_dir = data_dir + "datasets/"
test_dir = save_dir + "test_data.h5"
weights_dir = save_dir + "weights_vnet.h5"

x_test, y_test = load_dataset(test_dir)
model = vnet(input_size=(256, 256, 32, 1))#64
model.load_weights(weights_dir)

predictions = predict(x_test, model)

predictions = np.array(np.split(predictions, 8, axis=0))

write_predictions(predictions, save_dir+"predictions_vnet.h5")
