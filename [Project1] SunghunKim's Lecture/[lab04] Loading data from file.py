import numpy as np
import tensorflow as tf

# x_data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96,98, 100], [73, 66, 70]]
# y_data = [[152], [185], [180], [196], [142]]

# current_directory = os.path.dirname(os.path.realpath(__file__))
file_name = 'data-01-test-score.csv'
# file_path = os.path.join(current_directory, file_name)

xy = np.loadtxt(file_name, delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

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

#launch the graph in a session
sess= tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                        feed_dict={X : x_data, Y : y_data })
    if(step % 100 == 0):
        print(step, "cost : ", cost_val, "\nPrediction:", hy_val)


# Ask my score
print("your score will be : ", sess.run(hypothesis, 
    feed_dict= {X: [[100,70,101]]} ))

print("Other scores will be : ", sess.run(hypothesis, 
    feed_dict={X: [[60, 70, 110], [90, 100, 80]]} ))