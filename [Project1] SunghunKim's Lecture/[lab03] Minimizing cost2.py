import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [1,2,3]

X=tf.placeholder(tf.float32, shape=[None])
Y=tf.placeholder(tf.float32, shape=[None])
W=tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = X * W
cost = tf.reduce_sum(tf.square(hypothesis-Y))

# Minimize : Gradient Descent using derivative:
# W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y)*X)
descent = W-learning_rate*gradient
update = W.assign(descent)

sess= tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
