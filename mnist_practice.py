import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
# Print version to verify tf is installed
print(tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x = images, y = numbers
#sns.countplot(x=y_train)
#plt.show()

#Check to make sure there are no values that are NAN (Not a Number)

print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing:", np.isnan(x_test).any())

#tell the model what shape to expect
input_shape = (28, 28, 1) #28x28 pixels, 1 color channel (grayscale)

#Reshape the Data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train/255.0 #Normalize the data to be between 0 and 1
#same for testing
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test/255.0 #Normalize the data to be between 0 and 1

#Convert our labels to be ONE-HOT, not sparse
y_train = tf.one_hot(y_train.astype(np.int32), depth = 10)
y_test = tf.one_hot(y_test.astype(np.int32), depth = 10)

#Show an example image from Mnist
plt.imshow(x_train[random.randint(0, 59999)][:,:,0], cmap= "gray")
plt.show()

batch_size = 128
num_classes = 10
epochs = 5

#Build the Model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5,5), padding = 'same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = input_shape),
        tf.keras.layers.Dense(num_classes, activation = 'softmax')
    ]
)

model.compile(optimizer = tf.keras.optimizers.RMes epsilon = 1e-60  'categorical_cross_entropy', )