import random
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['MKL_NUM_THREADS'] = '16'
#os.environ['GOTO_NUM_THREADS'] = '16'
#os.environ['OMP_NUM_THREADS'] = '16'
#os.environ['openmp'] = 'True'

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras import regularizers

from sklearn.model_selection import train_test_split

def run_training(num_classes,batch_size,epochs,num_nodes,activation,optimizer,regularizer,lr,momentum,
                    lmbda,loss,patienceLR,patienceEarlyStopping,x_train,y_train, x_train_unlabeled):
            start = time.time()
            print('started')


            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

            y_sol = test[:,0]
            x_sol = test[:,1:]

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            model = Sequential()
            model.add(Dense(num_nodes,activation=activation,input_shape=(128,),
                            kernel_regularizer=regularizers.l2(lmbda),))

            model.add(Dense(num_nodes,activation=activation))

            model.add(Dense(num_classes, activation='softmax'))

            sgd = SGD(lr=lr,momentum=momentum)
            model.compile(loss=loss,
                          optimizer=sgd,
                          metrics=['accuracy'])

            learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                        patience=patienceLR,
                                                        verbose=1,
                                                        factor=0.5,
                                                        min_lr=0.0001)

            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(x_test, y_test),
                                #callbacks=[learning_rate_reduction])
                                callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-6,patience=patienceEarlyStopping, verbose=1, mode='auto'),learning_rate_reduction])

            loss, accuracy = model.evaluate(x_test, y_test)
            print('\nLoss is:')
            print(loss)
            print('2000 Epochs and hidden nodes: ')
            print(num_nodes)
            print('Lamda is: ')
            print(lmbda)
            print('Accuracy is:')
            print(accuracy)

            sol = model.predict(test[:,1:], batch_size=None, verbose=0)

            n = sol.shape[0]

            y_solL = np.zeros(n)
            y_solR = np.zeros(n)
            for i in range(0,n):
                y_solL[i] = 30000+i
                y_solR[i] = np.argmax(sol[i])
            y_sol = list(zip(y_solL,y_solR))

            df = pd.DataFrame(y_sol)
            df.to_csv("sol.csv",',',index=None,header=["Id","y"],float_format='%.0f')

            end = time.time()
            print('Time elapsed:')
            print(end-start)

lmbda = 0
num_classes = 10
batch_size = 100
epochs = 200
activation ='tanh' #activation for the first layer
optimizer ='sgd'
regularizer ='l2'
lr=0.01
momentum=0
num_nodes = 30
loss='categorical_crossentropy'
patienceLR = 20
patienceEarlyStopping = 100

train_labeled = pd.read_csv("train_labeledCSV.csv")
train_unlabeled = pd.read_csv("train_unlabeledCSV.csv")
test = pd.read_csv("testCSV.csv")

train_labeled = train_labeled.values
train_unlabeled = train_unlabeled.values
test = test.values

y_train_labeled = train_labeled[:9000,1]
x_train_labeled = train_labeled[:9000,2:]
x_train_unlabeled = train_unlabeled[:21000,:]
#y_train_labeled = train[:1000,1]
#x_train_labeled = train[:1000,2:]
#x_train_unlabeled = x_train_unlabeled[:1000,:]

'''
for ii in range(0,layerSizes.size):
    num_nodes = layerSizes[ii]
    run_training(num_classes,batch_size,2000,num_nodes,activation,optimizer,regularizer,lr,momentum,0,loss,patienceLR,patienceEarlyStopping,x_train,y_train)
'''

run_training(num_classes,batch_size,epochs,num_nodes,activation,optimizer,regularizer,lr,momentum,lmbda,loss,patienceLR,patienceEarlyStopping,x_train_labeled,y_train_labeled, x_train_unlabeled)