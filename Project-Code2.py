#Initializations and loading packages

import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from keras import backend as K
from keras import optimizers

from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dense, Activation

NEpochs = 1000
BatchSize=250
Optimizer=optimizers.RMSprop(lr=0.01)

def SetTheSeed(Seed):
    np.random.seed(Seed)
    rn.seed(Seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    tf.set_random_seed(Seed)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

#%% Loading data and splitting them into X and Y

TrainData = pd.read_csv('train_1_dum.csv',sep=',',header=0,quotechar='"')
ValData = pd.read_csv('valid_1_dum.csv',sep=',',header=0,quotechar='"')

y_train = np.array(TrainData['click'])
x_train = np.array(TrainData.iloc[:,1:])

y_valid = np.array(ValData['click'])
x_valid = np.array(ValData.iloc[:,1:])

#%% Using the softmax approach
SetTheSeed(11)

nn = Sequential()

nn.add(Dense(units=3,input_shape=(x_train.shape[1],),activation="relu",use_bias=True))
nn.add(Dense(units=3,activation="relu",use_bias=True))
nn.add(Dense(units=3,activation="relu",use_bias=True))
nn.add(Dense(units=3,activation="relu",use_bias=True))
nn.add(Dense(units=2,activation="softmax",use_bias=True))

nn.compile(loss='categorical_crossentropy', optimizer=Optimizer,metrics=['categorical_crossentropy'])

#%% Fitting NN Model with Softmax
y_train = np.array([1-y_train,y_train]).transpose()

FitHist = nn.fit(x_train,y_train,epochs=NEpochs,batch_size=BatchSize,verbose=0)

#%% Making Predictions

y_train_pred = nn.predict(x_train,batch_size=x_train.shape[0])
y_valid_pred = nn.predict(x_valid,batch_size=x_valid.shape[0])

#%% Calculating Log Loss
log_loss(y_train, y_train_pred)
log_loss(y_valid, y_valid_pred)
