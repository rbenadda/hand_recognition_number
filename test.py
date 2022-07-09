import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys


#assert hasattr(tf,"function")

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#targets_names = ["0","1","2","3","4","5","6","7","8","9"]
#plt.imshow(images[1000],cmap="binary")
#plt.title(targets_names[targets[1000]])
#plt.show()

#images_train, images_test,targets_train,targets_test = train_test_split(images,targets,test_size=0.2,random_state=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28,1)))
model.add(tf.keras.layers.Dense(784, activation="relu"))
model.add(tf.keras.layers.Dense(50, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(500, activation="relu"))
model.add(tf.keras.layers.Dense(1000, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

hist = model.fit(x_train, y_train,batch_size=128,epochs=20,verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('Model1.h5')
