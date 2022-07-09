# -*-coding:Latin-1 -*

    #Import
    # Matplotlib
import matplotlib.pyplot as plt
    # Tensorflow
import tensorflow as tf
    # Numpy and Pandas
import numpy as np
import pandas as pd
    # Ohter import
import sys

    #From
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

    #Assert
    # Etre sur d'utiliser tensorflow 2.0
assert hasattr(tf, "function") 


    # MNIST (base de données)
mnist = tf.keras.datasets.mnist
(images, targets),(_,_) = mnist.load_data()

    # Obtenir une sous-partie de l'ensemble des données MNIST
images = images[:10000]
targets = targets [:10000]
    #Donner un nom à chaque chiffre
targets_names = ["0","1","2","3","4","5","6","7","8","9"]

    # Dessiner l'image
plt.imshow(images[3], cmap="binary")
plt.title(targets_names[targets[3]])
plt.show()

    # Reformer les données en float
images = images.reshape(-1, 784)
images = images.astype(float)
#images_test = images_test.reshape(-1, 784)
#images_test = images_test.astype(float)
#print(targets.shape)

    #StandardScaler va transformer nos données, tels que sa distribution aura une valeur moyenne 0 et d'écart-type de 1.
#scaler = StandardScaler()
#images = scaler.fit_transform(images)
#images_test = scaler.transform(images_test)
images_train, images_test, targets_train, targets_test = train_test_split( images, targets, test_size=0.2, random_state=1 )

    #Pour afficher les dimentions
#print(images_train.shape, targets_train.shape)
#print(images_test.shape, targets_test.shape)


    # Créer un modèle aplati 
model = tf.keras.models.Sequential()

    # Ajout de couches
    #
model.add(tf.keras.layers.Dense(256, activation="sigmoid"))
model.add(tf.keras.layers.Dense(128, activation="sigmoid"))
    #Fait en sorte que la somme de toutes les activations soit égal à 1
model.add(tf.keras.layers.Dense(256, activation="softmax")) 

model_output = model.predict(images[0:1])
#print(model_output, targets[0:1])

model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

history = model.fit(images_train, targets_train, epochs=10, validation_split=0.2)

loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]
loss_val_curve = history.history["val_loss"]
acc_val_curve = history.history["val_accuracy"]

plt.plot(loss_curve, label="Train")
plt.plot(loss_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Loss")
plt.show()


plt.plot(acc_curve, label="Train")
plt.plot(acc_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Accuracy")
#plt.show()

#model.save('Model1.h5')
#loaded_model = tf.keras.load_model("Model1.h5")
#loaded_model.predict(images_test[0:1]), target[0:1]


loss, acc = model.evaluate(images_test, targets_test)
print("Test Loss", loss)
print("Test Accuracy", acc)
