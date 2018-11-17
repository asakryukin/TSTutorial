import tensorflow as tf
import json
import numpy as np
class Model:

    def __init__(self):
        self.sess = tf.Session()
        self.create_model()


    def create_model(self):

        if True:
            self.input_tensor = tf.placeholder(tf.float32, [None,28,28,1], name='input')
            self.keep_prob_tensor = tf.placeholder_with_default(tf.constant(1.0),tf.constant(1.0).shape)
            self.labels_tensor = tf.placeholder(tf.int64,[None])

            net = tf.layers.conv2d(self.input_tensor,8,[3,3],activation=tf.nn.relu, name='l1', reuse=False)
            net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],'SAME',name='l1max')
            net = tf.nn.dropout(net,keep_prob=self.keep_prob_tensor)

            net = tf.layers.conv2d(net, 16, [3, 3], activation=tf.nn.relu, name='l2')
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME',name='l2max')
            net = tf.nn.dropout(net, keep_prob=self.keep_prob_tensor)

            net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.relu, name='l3')
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME',name='l3max')
            net = tf.nn.dropout(net, keep_prob=self.keep_prob_tensor)

            net = tf.layers.flatten(net)

            net = tf.layers.dense(net, 100, activation=tf.nn.relu, name= 'l4')
            net = tf.nn.dropout(net, keep_prob=self.keep_prob_tensor)

            net = tf.layers.dense(net, 10, name='l5')

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels_tensor,10), logits=net))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(tf.nn.softmax(net),1),self.labels_tensor),tf.float32))
            self.scores = tf.nn.softmax(net,name='output')
            self.output = tf.arg_max(self.scores,1)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

            self.sess.run(tf.initialize_all_variables())

    def train(self,DH, N):
        self.sess.run(tf.initialize_all_variables())

        for i in range(N):
            data,labels = DH.get_training_batch(64)
            self.sess.run(self.optimizer, {self.input_tensor: data, self.labels_tensor: labels, self.keep_prob_tensor: 0.5})

            if i%1000 == 0:
                loss, accuracy, o = self.sess.run([self.loss, self.accuracy, self.output],{self.input_tensor: DH.valid_data, self.labels_tensor: DH.valid_labels})
                print("Iteration "+str(i)+" loss: "+str(loss)+" accuracy: "+str(accuracy)+" output: "+str(o))

        self.saver = tf.saved_model.simple_save(self.sess, 'Model', {'input': self.input_tensor},
                                                        {'output': self.output})

        for n in tf.get_default_graph().as_graph_def().node:
            print(n.name)

        with open('TensorSpace/mnist/data/data.json', 'w') as outfile:
            outfile.write(str(list(DH.valid_data[0].ravel())))


