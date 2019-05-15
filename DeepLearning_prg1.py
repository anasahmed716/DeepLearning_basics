# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:46:35 2019

@author: AnasAhmed
"""

import tensorflow as tf

mnist = tf.keras.datasets.mnist
mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

"""print(x_train[0])
x_train.shape
x_train.size
"""

import matplotlib.pyplot as plt 

plt.imshow(x_train[0])
#plt.imshow(x_train[10000])
#plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

#%%Building Model

model = tf.keras.models.Sequential()

#1. Input Layer
"""It's going to take the data we throw at it, and just flatten it for us."""

model.add(tf.keras.layers.Flatten())                        #our image is 28x28...we need to flatten it

#2. Hidden Layer
""" We're going to go with the simplest neural network layer,
which is just a Dense layer. This refers to the fact that it's a densely-connected layer, 
meaning it's "fully connected," where each node connects to each prior and subsequent node.
Just like our image."""
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))   #128 neurons in the layer.
"""This layer has 128 units. The activation function is relu, short for rectified linear. 
Currently, relu is the activation function you should just default to. 
There are many more to test for sure, but, if you don't know what to use, 
use relu to start.

Let's add another identical layer for good measure."""
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#3. Output Layer
"""This is our final layer. It has 10 nodes. 1 node per possible number prediction. 
In this case, our activation function is a softmax function, 
since we're really actually looking for something more like a probability distribution 
of which of the possible prediction options this thing we're passing features through 
of is. Great, our model is done."""
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

"""Now we need to "compile" the model. This is where we pass the settings for actually 
optimizing/training the model we've defined."""

model.compile(optimizer='adam',                         # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',   # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])                     # what to track

"""Remember why we picked relu as an activation function? 
Same thing is true for the Adam optimizer. It's just a great default to start with.

Next, we have our loss metric. Loss is a calculation of error. 
A neural network doesn't actually attempt to maximize accuracy. 
It attempts to minimize loss. Again, there are many choices, 
but some form of categorical crossentropy is a good start for a classification task 
like this.

Now, we fit!"""

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy

#%%To save and load the model

#model.save('epic_num_reader.model')
#new_model = tf.keras.models.load_model('epic_num_reader.model')

#%%Predicting

predictions = model.predict(x_test)
print(predictions)

import numpy as np

print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()




