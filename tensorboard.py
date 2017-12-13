import tensorflow as tf
import os

session = tf.Session()
x = tf.placeholder(tf.float32, name='alpha')
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
b = tf.Variable(tf.zeros([1]), name='biases')

linear_model = W * x + b
init = tf.global_variables_initializer()
session.run(init)

file_writer = tf.summary.FileWriter('{}/tf.log'.format(os.getcwd()), session.graph)