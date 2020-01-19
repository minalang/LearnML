import os
import numpy as np
import tensorflow as tf

# x_data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96,98, 100], [73, 66, 70]]
# y_data = [[152], [185], [180], [196], [142]]

file_names = ['data-01-test-score.csv', 'data-02-test-score.csv']

file_name_queue = tf.train.string_input_producer(file_names, shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(file_name_queue)

# Default value, in case of empty columns.
record_defaults = [ [0.] for _ in range(4) ]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# Collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size = 3)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis
hypothesis = tf.matmul(X, W) + b
# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train  = optimizer.minimize(cost)

# Launch the graph in a session
sess= tf.Session()
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess= sess, coord=coord)


for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                        feed_dict={X : x_batch, Y : y_batch })
    #print(x_batch, y_batch)
    if(step % 100 == 0):
        print(step, "cost : ", cost_val, "\nPrediction:", hy_val)

coord.request_stop()
coord.join(threads)

# Ask my score
print("your score will be : ", sess.run(hypothesis, 
    feed_dict= {X: [[100,70,101]]} ))

print("Other scores will be : ", sess.run(hypothesis, 
    feed_dict={X: [[60, 70, 110], [90, 100, 80]]} ))