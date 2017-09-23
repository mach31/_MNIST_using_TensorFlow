# ==============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#impporting the mnist dataset with one_hot enabled
mnist = input_data.read_data_sets("add path of MNIST dataset", one_hot = True)

#number of nodes at each hidden layers
nodes_of_hl1 = 500
nodes_of_hl2 = 500
nodes_of_hl3 = 500
num_of_classes = 10
batch_size = 100

x = tf.placeholder(tf.float32, [None, 784], name = 'x_data') #to store data
y = tf.placeholder(tf.float32, name = 'y_data')# to store labels

# here's our computational graph
def neural_net_model(data):
    hidden_layer1 = {'weights' : tf.Variable(tf.random_normal([784, nodes_of_hl1]), name = 'weight0_1'), 'biases' : tf.Variable(tf.random_normal([nodes_of_hl1]), name = 'biases0_1') }
    hidden_layer2 = {'weights' : tf.Variable(tf.random_normal([nodes_of_hl1, nodes_of_hl2]), name = 'weight1_2'), 'biases' : tf.Variable(tf.random_normal([nodes_of_hl2]), name = 'biases1_2') }
    hidden_layer3 = {'weights' : tf.Variable(tf.random_normal([nodes_of_hl2, nodes_of_hl3]), name = 'weight2_3'), 'biases' : tf.Variable(tf.random_normal([nodes_of_hl3]), name = 'biases2_3') }
    output_layer = {'weights' : tf.Variable(tf.random_normal([nodes_of_hl3, num_of_classes]), name = 'weight3_out'), 'biases' : tf.Variable(tf.random_normal([num_of_classes]), name = 'biases3_out') }

    """ what's happening at each layers...
        weights are multiplied with data from each previous layer nodes and so on.
        And the result is added with a bias before this result is fed to the next layer nodes.
    """

    at_layer1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    at_layer1 = tf.nn.relu(at_layer1)
    at_layer2 = tf.add(tf.matmul(at_layer1, hidden_layer2['weights']), hidden_layer2['biases'])
    at_layer2 = tf.nn.relu(at_layer2)
    at_layer3 = tf.add(tf.matmul(at_layer2, hidden_layer3['weights']), hidden_layer3['biases'])
    at_layer3 = tf.nn.relu(at_layer3)

    output = tf.add(tf.matmul(at_layer1, output_layer['weights']), output_layer['biases'])

    return output

pred = neural_net_model(x)
lr = 0.01
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
num_of_epochs = 10
init = tf.global_variables_initializer()
saver = tf.train.Saver()

print('\n*****************************************************************\n')
print('1st session of training and testing of save operations in TF.')
print('\n*****************************************************************\n')
print("Total Number of Epochs:", num_of_epochs, "\n")

with tf.Session() as sess:
    sess.run(init)
    #to find elapsed time of the program
    import time
    st_time = time.time()

    print("Number of Epochs To Be Performed Now:", num_of_epochs//2)

    for epoch in range(num_of_epochs//2):
        loss = 0
        for _ in range(int(mnist.train.num_examples/batch_size)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            loss += c
        print("Epoch", epoch, "With Loss: ", format(loss, "0.4f"))
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy = accuracy.eval({x: mnist.test.images, y : mnist.test.labels})*100

    print("Accuracy: ", format(accuracy, "0.4f"), "%")
    print("Elapsed Time Is %s seconds" %(format((time.time() - st_time), "0.4f")))

    save_path = saver.save(sess, "add path to store the model")
    print("Model saved in file: %s" % save_path)

print('\n*****************************************************************\n')
print('2nd session of training and testing of save operations in TF.')
print('\n*****************************************************************\n')
print("Total Number of Epochs: ", num_of_epochs, "\n")

with tf.Session() as sess:
    sess.run(init)

    import time
    st_time = time.time()

    saver.restore(sess, "add path of the stored network")
    print("\nModel restored from file: %s" % save_path)
    print("Number of Epochs To Be Performed Now: ", num_of_epochs//2, "\n")
    for epoch in range(num_of_epochs//2):
        loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for _ in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            loss += c
        print("Epoch", epoch, "With Loss : ", format(loss, "0.4f"))

    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy = accuracy.eval({x: mnist.test.images, y : mnist.test.labels})*100

    print("Accuracy: ", "{:.4f}".format(accuracy), "%")
    print("Elapsed Time Is %s seconds" %(format((time.time() - st_time), "0.4f")))
