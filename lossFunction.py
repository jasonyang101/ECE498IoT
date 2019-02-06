import tensorflow as tf
import numpy as np
def function(a,X,b,y):
    t_X = tf.transpose(X)
    t_b = tf.transpose(b)
    return (a*tf.matmul(t_X,X)+tf.matmul(t_b,X)-y)**2
