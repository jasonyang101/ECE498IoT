from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# get data and normalize
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
# print(test_labels.shape)

a = np.array(train_labels)
b = np.zeros((60000,10))
b[np.arange(len(a)),a] = 1
train_labels = b

a = np.array(test_labels)
b = np.zeros((10000,10))
b[np.arange(10000),a] = 1
test_labels = b

learning_rate = 0.001
num_steps = 1875
batch_size = 32
display_step = 32
# me defined
total_vals = 60000
curr_rand = np.arange(60000)
epochs = 10
# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, 28, 28])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, padd, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padd)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def next_batch(train_images, train_labels, current):
    if current+batch_size >= total_vals:
        new_vals = curr_rand[current:]
        np.random.shuffle(curr_rand)
        current = -batch_size
    else:
        new_vals = curr_rand[current:current+batch_size]
    batch_x, batch_y = [], []
    for idx in new_vals:
        batch_x.append(train_images[idx])
        batch_y.append(train_labels[idx])
    return batch_x, batch_y, current+batch_size

# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], "VALID")
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], "SAME")
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.softmax(fc3)
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)
    return fc3

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 3 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 3], stddev=0.1)),
    # 3x3 conv, 3 inputs, 3 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 3, 3], stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([108, 100], stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'wd2': tf.Variable(tf.random_normal([100, 50], stddev=0.1)),
    # 3rd dense layer
    'wd3': tf.Variable(tf.random_normal([50, 10], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([3])),
    'bc2': tf.Variable(tf.random_normal([3])),
    'bd1': tf.Variable(tf.random_normal([100])),
    'bd2': tf.Variable(tf.random_normal([50])),
    'bd3': tf.Variable(tf.random_normal([10])),
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    current = 0
    loss_out = []
    acc_out = []
    for i in range(epochs):
        print("EPOCH NUMBER: " + str(i))
        for step in range(1, num_steps+1):
            batch_x, batch_y, current = next_batch(train_images, train_labels, current)
            # print(batch_x.shape,end="")
            # print(batch_y.shape)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if step % 25 == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        print("epoch done")
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: test_images,
                                                             Y: test_labels,
                                                             keep_prob: 1.0})
        print("Loss: " + "{:.4f}".format(loss))
        loss_out.append("{:.4f}".format(loss))
        print("Acc: "+"{:.3f}".format(acc))
        acc_out.append("{:.3f}".format(acc))
    print(loss_out)
    print(acc_out)


    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_images,
                                      Y: test_labels,
                                      keep_prob: 1.0}))
