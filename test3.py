import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

assert hasattr(tf,"function")


mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()




x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

        # convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)
        
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape=(28,28,1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation="relu", input_shape=input_shape))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adadelta",metrics=["accuracy"])

hist = model.fit(x_train, y_train,batch_size=128,epochs=20,verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('Modeltest.h5')
