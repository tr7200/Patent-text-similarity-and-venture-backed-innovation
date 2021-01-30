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
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import regularizers, optimizers
import keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt



np.random.seed(1)


# Reproducibility code

# save python environment when run
open('Keras_patent_text_training_environment-2-14-20.txt', 
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
pickle.dump(config, 
            open('Keras_patent_text_training_config-2-14-20.p', 'wb'))


# Using AMD gpu with PlaidML and Metal
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# Create our data function

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
                "Patent_text_cosine_similarity_training_MinMaxScaler-2-14-20.save")
    
    return train_x, train_y, test_x, test_y


# Load the data
FEATURES = '../Patent_text_independent_variables-11-13-19.csv'
TARGET = '../Patent_count_y-11-13-19.csv'

train_x, train_y, test_x, test_y = data(FEATURES, TARGET)


def patent_value_loss(y_true, y_pred):
  '''Custom loss metric for patent values
  
  Args:
      y_true: ground truth
      y_pred: predictions
      
  Returns:
      Loss estimated in USD baased on value of patent from:
      https://www.ipwatchdog.com/2017/07/12/patent-portfolio-valuations/id=85409/
  '''
  patent_value_loss = K.abs(1 - K.exp(y_true - y_pred)) * 50000
    
  return patent_value_loss


"""
Instantiate a model - the architectural choices chosen by Hyperas
led to erratic loss during training, this is slightly different

The loss function is MSE, which is more appropriate for a
logarithmic dependent variable, but MAE is added as a metric as 
well since it is robust to outliers
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

model.compile(loss='mean_squared_error', 
              metrics=['mae', 
                       patent_value_loss],
              optimizer='adam')

callbacks = [EarlyStopping(monitor='val_loss',
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
             ModelCheckpoint("patent_text_model_epoch_no.{epoch:03d}-2-14-20.h5",
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',
                             period=1),
             CSVLogger('patent_text_training-2-14-20.log')]

result = model.fit(train_x,
                   train_y,
                   batch_size=16,
                   epochs=30,
                   verbose=1,
                   callbacks=callbacks,
                   validation_split=0.05)


# MSE loss plot
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# MAE metric plot
plt.plot(result.history['mean_absolute_error'])
plt.plot(result.history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# patent value loss plot
plt.plot(result.history['patent_value_loss'])
plt.plot(result.history['val_patent_value_loss'])
plt.title('model patent value loss')
plt.ylabel('patent value loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


model = load_model('patent_text_model_epoch_no.030-2-14-20.h5', 
                   custom_objects={'patent_value_loss': patent_value_loss})

print(model.evaluate(test_x, test_y))
