from __future__ import print_function
import tensorflow as tf
import os

W = tf.Variable([.3], dtype=tf.float32)         # weight
b = tf.Variable([-.3], dtype=tf.float32)        # bias
x = tf.placeholder(tf.float32)
model = W * x + b
y = tf.placeholder(tf.float32)

# calculate loss
# cost function J(theta0, theta1) = 1/2m sum((i=1 to m) (h(x(i))-y(i))2)
# theta0 -  Weight
# theta1 - bias

loss = tf.reduce_sum(tf.square(model-y))

optimizer = tf.train.GradientDescentOptimizer(0.001)         # learning rate (alpha) = 0.01

train = optimizer.minimize(loss)            # minimize the cost function

# train the model

x_trainingData = list(range(400, 10000, 100))
y_trainingData = list(range(200, 5000, 50))
x_trainingData = [a/4800.0 for a in x_trainingData]
y_trainingData = [a/2400.0 for a in y_trainingData]
print (x_trainingData)
print (y_trainingData)
# x_trainingData=[1,2,3,4]
# y_trainingData=[0,-1,-2,-3]
init=tf.global_variables_initializer() # To initialize all Variable
tf.summary.histogram("loss", loss)
sess=tf.Session()
file_writer = tf.summary.FileWriter('{}/histogram'.format(os.getcwd()))
# summaries = tf.summary.merge_all()
sess.run(init) # Run with initial values

for i in range(10000):
    print (sess.run(train, feed_dict={x: x_trainingData, y: y_trainingData}))
    # file_writer.add_summary(summ, global_step=i)

# calculate training accuracy
curr_Weight, curr_bias, curr_loss = sess.run([W, b, loss],feed_dict={x:x_trainingData, y:y_trainingData})

print ("W: %s b: %s loss: %s"%(curr_Weight, curr_bias, curr_loss))
# file_writer = tf.summary.FileWriter('{}/tf.log'.format(os.getcwd()), sess.graph)
# hist = tf.summary.histogram('histogram', curr_loss)

