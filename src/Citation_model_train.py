#!/usr/bin/env python
# coding: utf-8

# Citation count training


from __future__ import print_function

import os

import subprocess
import platform
import datetime as dt
import pickle, joblib

import numpy as np
from pandas import read_csv

import keras
from keras.callbacks import (ModelCheckpoint, 
                             CSVLogger, 
                             EarlyStopping, 
                             ReduceLROnPlateau)
from keras.layers.core import Dense, Activation
from keras.layers import PReLU
from keras.models import Sequential
from keras.utils import np_utils
from keras import regularizers, optimizers
from keras.models import load_model
import keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt



def data(indep_variables, target):
    """Import data, drop ids, scale data to 0 and 1 for activation functions"""
    indep_variables = read_csv(indep_variables, header = 0)
    target = read_csv(target, header = 0)
    train_x, test_x, train_y, test_y= train_test_split(indep_variables, 
                                                       target, 
                                                       test_size = 0.05, 
                                                       random_state = 1)
    x_scaler = MinMaxScaler().fit(train_x)
    train_x = x_scaler.transform(train_x)
    test_x = x_scaler.transform(test_x)
    joblib.dump(x_scaler, 
                "Citations_text_cosine_similarity_training_MinMaxScaler-2-17-20.save")
    
    return train_x, train_y, test_x, test_y


def get_model():
    """
    Instantiate a model - the architectural choices chosen by Hyperas
    led to erratic loss during training, this is slightly different

    Loss function is MSE, which is more appropriate for a
    logarithmic dependent variable though not as robust to outliers
    as MAE, therefore MAE is added as a metric as well.
    """

    model = Sequential()
    model.add(Dense(31, input_shape=(31,), kernel_initializer='he_normal'))
    model.add(keras.layers.PReLU())
    model.add(Dense(26, kernel_initializer='he_normal'))
    model.add(keras.layers.PReLU())
    model.add(Dense(24, kernel_initializer='he_normal'))
    model.add(keras.layers.PReLU())
    model.add(Dense(20, kernel_initializer='he_normal'))
    model.add(keras.layers.PReLU())
    model.add(Dense(18, kernel_initializer='he_normal'))
    model.add(keras.layers.PReLU())
    model.add(Dense(14, kernel_initializer='he_normal'))
    model.add(keras.layers.PReLU())
    model.add(Dense(6, kernel_initializer='he_normal'))
    model.add(keras.layers.PReLU())
    model.add(Dense(8, kernel_initializer='he_normal'))
    model.add(keras.layers.PReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))

    # MSE and MAE
    model.compile(loss='mean_squared_error', 
                  metrics=['mae'], 
                  optimizer='adam')
    
    return model


def main(features: str=INDEP_VARIABLES,
         target: str=TARGET,
         callbacks: List=CALLBACKS):
         """ Load the data, create model, train"""

         train_x, train_y, test_x, test_y = data(features, target)
     
         model = get_model()
    
         result = model.fit(train_x,
                            train_y,
                            batch_size=16,
                            epochs=30,
                            verbose=1,
                            callbacks=callbacks,
                            validation_split=0.05)
          return result


if __name__ == '__main__':
    # Reproducibility code

    # save python environment when run
    open('Keras_citations_text_training_environment-2-17-20.txt', 
         'wb').write(subprocess.check_output(['pip', 'list']))

    # underlying platform
    system = platform.uname()

    # python version
    python_version = platform.python_version()

    # date and time of run
    date = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    config = {
        'system': system,
        'python_version': python_version,
        'date': date
    }

    pickle.dump(config, open('Keras_citations_text_training_config-2-17-20.p', 'wb'))
  
  
    np.random.seed(1)
    # Using AMD gpu with PlaidML and Metal
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


    # The data and callbacks
    INDEP_VARIABLES = '../Patent_text_independent_variables-11-13-19.csv'
    TARGET = '../Citation_count_y-11-13-19.csv'

    CALLBACKS = [EarlyStopping(monitor='val_loss',
                           patience=3,
                           verbose=1,
                           mode='min',
                           restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=2,
                               verbose=1,
                               mode='auto',
                               min_lr=1e-5),
                 ModelCheckpoint("citations_text_model_epoch_no.{epoch:03d}-2-17-20.h5",
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',
                             period=1),
                 CSVLogger('citations_text_training-2-17-20.log')]
  
  
    # Get data and fit model
    result = main(features=INDEP_VARIABLES,
                  target=TARGET,
                  callbacks=CALLBACKS)
    
    
    # Loss plot
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


    # MAE plot
    plt.plot(result.history['mean_absolute_error'])
    plt.plot(result.history['val_mean_absolute_error'])
    plt.title('model MAE')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


    """
    As with the training for patent counts, this training was erratic 
    and also started overfitting very quickly. The width and depth of 
    the layers that hyperas chose worked well but the adaptive optimizer 
    could not handle the data and a learning rate reduction schedule was 
    necessary. In retrospect, I may have used a learning rate range test 
    to find a good learning rate.
    """

    model1 = load_model('citations_text_model_epoch_no.029-2-17-20.h5')
    print(model1.evaluate(test_x, test_y))

    """
    The loss and MAE of this epoch of training are 0.79 and 0.68, 
    respectively, much higher than those for the patent count training 
    but most of the patents had at least a couple of citations. Since 
    the margin of error is around 2 patents (these are logs), pay 
    attention to these results if they start predicting 3 or more 
    citations for a patent based on the description text simiarity 
    with the patents of peer companies--that indicates quality, which 
    leads to value.
    """
