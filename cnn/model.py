# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages 


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.0015,
                 learning_rate_decay_factor=0.90):
        self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        x = tf.reshape(self.x_, [-1, 28, 28, 1])

        # TODO: implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss
        #        the 10-class prediction output is named as "logits"

        #第一个卷积
        W_conv1 = weight_variable([5, 5, 1, 16])
        b_conv1 = bias_variable([16])

        h_conv1 = conv2d(x, W_conv1) + b_conv1
        h_bn1 = batch_normalization_layer(h_conv1, isTrain = is_train)

        h_relu1 = tf.nn.relu(h_bn1)
        h_pool1 = max_pool_2x2(h_relu1)

        #第二个卷积
        W_conv2 = weight_variable([5, 5, 16, 32])
        b_conv2 = bias_variable([32])

        h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
        h_bn2 = batch_normalization_layer(h_conv2, isTrain = is_train)

        h_relu2 = tf.nn.relu(h_bn2)
        h_pool2 = max_pool_2x2(h_relu2)

        #Linear
        W_fc1 = weight_variable([7 * 7 * 32, 10])
        b_fc1 = bias_variable([10])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
        h_fc1_drop = tf.nn.dropout(h_pool2_flat, self.keep_prob)
        logits = tf.matmul(h_fc1_drop, W_fc1) + b_fc1

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    EPSILON = 0.001
    CHANNEL = inputs.shape[3]
    MEANDECAY = 0.99

    ave_mean = tf.Variable(tf.zeros(shape = [CHANNEL]), trainable = False)
    ave_var = tf.Variable(tf.zeros(shape = [CHANNEL]), trainable = False)

    mean, var = tf.nn.moments(inputs, axes = [0, 1, 2], keep_dims = False)

    update_mean_op = moving_averages.assign_moving_average(ave_mean, mean, MEANDECAY)
    update_var_op = moving_averages.assign_moving_average(ave_var, var, MEANDECAY)

    tf.add_to_collection("update_op", update_mean_op)
    tf.add_to_collection("update_op", update_var_op)
    
    scale = tf.Variable(tf.constant(1., shape = mean.shape))
    offset = tf.Variable(tf.constant(0., shape = mean.shape))

    if isTrain:
        inputs = tf.nn.batch_normalization(inputs, mean = mean, variance = var, offset = offset, scale = scale, variance_epsilon = EPSILON)

    else :
        inputs = tf.nn.batch_normalization(inputs, mean = ave_mean, variance = ave_var, offset = offset, scale = scale, variance_epsilon = EPSILON)

    return inputs

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
