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

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=5, padding="valid", activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding="same", activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_images, train_labels, epochs=10)
model.summary()
model.save("keras_model.h5")


# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)



#4d weight matrix
