# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.008,
                 learning_rate_decay_factor=0.95):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        # W_fc1 = weight_variable([28 * 28, 10])
        # b_fc1 = bias_variable([10])

        # #x_image = tf.reshape(self.x_, [-1,28,28,1])
        # print(self.x_.shape)
        # logits = tf.matmul(self.x_, W_fc1) + b_fc1
        # print(logits.shape)
        # h_fc1 = tf.nn.relu(tf.matmul(self.x_, W_fc1) + b_fc1)

        # y = tf.nn.softmax(h_fc1)

        # self.W1 = weight_variable(shape = [784, 1024])
        # self.b1 = bias_variable(shape = [1024])

        # self.u1 = tf.matmul(self.x_, self.W1) + self.b1
        # self.y1 = tf.nn.relu(self.u1)

        # logits = self.y1

        self.W1 = weight_variable(shape = [784, 1024])
        self.b1 = bias_variable(shape = [1024])

        self.u1 = tf.matmul(self.x_, self.W1) + self.b1
        self.y1 = tf.nn.relu(self.u1)

        self.W2 = weight_variable(shape = [1024, 300])
        self.b2 = bias_variable(shape = [300])

        self.u2 = tf.matmul(self.y1, self.W2) + self.b2
        self.y2 = tf.nn.relu(self.u2)

        self.W3 = weight_variable(shape = [300, 10])
        self.b3 = bias_variable(shape = [10])
        logits = tf.matmul(self.y2, self.W3) + self.b3

        #logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers
        #logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on fully-connected layers
    return inputs



