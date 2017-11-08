# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages 


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.0015,
                 learning_rate_decay_factor=0.92):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        W_fc1 = weight_variable([28 * 28, 1000])
        b_fc1 = bias_variable([1000])

        h_fc1 = tf.matmul(self.x_, W_fc1) + b_fc1

        # perform batch-normalization
        h_bn1 = batch_normalization_layer(h_fc1, isTrain = is_train)

        h_relu1 = tf.nn.relu(h_bn1)

        w_fc2 = weight_variable([1000, 600])
        b_fc2 = bias_variable([600])

        h_fc2 = tf.matmul(h_relu1, w_fc2) + b_fc2

        h_relu2 = tf.nn.relu(h_fc2)

        h_fc2_drop = tf.nn.dropout(h_relu2, self.keep_prob)

        w_fc3 = weight_variable([600, 10])
        b_fc3 = bias_variable([10])

        logits = tf.matmul(h_fc2_drop, w_fc3) + b_fc3

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
    EPSILON = 0.001
    SHAPES = inputs.shape[1]
    MEANDECAY = 0.99

    ave_mean = tf.Variable(tf.zeros(shape = [1, SHAPES]), trainable = False)
    ave_var = tf.Variable(tf.zeros(shape = [1, SHAPES]), trainable = False)

    inputs_shape = inputs.get_shape() 
    axis = list(range(len(inputs_shape) - 1))
    mean, var = tf.nn.moments(inputs, axes = axis, keep_dims = True)

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