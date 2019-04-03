import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=3, kernel_size=3, padding="valid", activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.DepthwiseConv2D(kernel_size=3,padding='same',depth_multiplier=1,strides=(1,1),use_bias=False))
model.add(keras.layers.Conv2D(filters=3,kernel_size=1,padding='valid',use_bias=False,strides=(1, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_images, train_labels, epochs=5)
model.summary()
model.save("keras_modified_1_3_1.h5")


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)



#4d weight matrix
