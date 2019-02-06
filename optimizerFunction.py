import tensorflow as tf
def function(loss,lr):
    return tf.train.AdamOptimizer(lr)
