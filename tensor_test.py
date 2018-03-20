from __future__ import print_function
import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

print (node1, node2)

session = tf.Session()
print (session.run([node1, node2]))

node3 = tf.add(node1, node2)
print ("node3:", node3)
print ("session.run(node3):", session.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print (session.run(adder_node, {a: 3, b: 4.5}))
print (session.run(adder_node, {a: [1, 3], b: [2, 4]}))

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
session.run(init)

