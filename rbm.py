import numpy as np
import tensorflow as tf


class RBM:

    def __init__(self, visible_nodes, hidden_nodes):
        # The weight matrix that stores the edge weights
        self.weights = tf.Variable(tf.random_normal([visible_nodes, hidden_nodes], stddev=0.1), name="weights")
        # The bias vector for the visible layer
        self.visible_bias = tf.Variable(tf.zeros([visible_nodes], dtype='float32'), name="visible_bias")
        # The bias vector for the hidden layer
        self.hidden_bias = tf.Variable(tf.zeros([hidden_nodes], dtype='float32'), name="hidden_bias")

    def sample(self, probabilities):
        return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities), minval=0, maxval=1), name="sample")

    def forward_propagation(self, v):
        return tf.sigmoid(tf.matmul(v, self.weights) + self.hidden_bias)

    def back_propagation(self, h):
        return tf.sigmoid(tf.matmul(h, tf.transpose(self.weights)) + self.visible_bias)

    def gibbs_sample(self, steps, h):
        v1 = np.empty([2,])
        h1 = np.empty([2,])
        for i in range(steps):
            v1 = self.sample(self.back_propagation(h))
            h1 = self.sample(self.forward_propagation(v1))

        return v1, h1

    def contrastive_divergence(self, x, steps, alpha):
        v0 = x
        h0 = self.sample(self.forward_propagation(v0))
        w_positive_gradient = tf.matmul(tf.transpose(v0), h0,
                                        name="positive_gradient")
        v1, h1 = self.gibbs_sample(steps=steps, h=h0)
        w_negative_gradient = tf.matmul(tf.transpose(v1), h1,
                                        name="negative_gradient")
        batch_size = tf.to_float(tf.shape(x)[0])
        weight_grad = tf.multiply(alpha, tf.subtract(w_positive_gradient, w_negative_gradient)/batch_size,
                                  name="weight_gradient")
        visible_bias_grad = tf.multiply(alpha, tf.reduce_sum(tf.subtract(v0, v1), axis=0)/batch_size,
                                        name="visible_bias_gradient")
        hidden_bias_grad = tf.multiply(alpha, tf.reduce_sum(tf.subtract(h0, h1), axis=0)/batch_size,
                                       name="hidden_bias_gradient")

        mse = tf.reduce_mean(tf.square(tf.subtract(v0, v1, name="v0_v1"), name="square_v0_v1"), name="mse")

        return weight_grad, visible_bias_grad, hidden_bias_grad, mse

    def fit(self, input_data, alpha):
        weight_grad, visible_bias_grad, hidden_bias_grad, mse = self.contrastive_divergence(x=input_data,steps=1,
                                                                                            alpha=alpha)
        return [self.weights.assign_add(weight_grad),
        self.visible_bias.assign_add(visible_bias_grad),
        self.hidden_bias.assign_add(hidden_bias_grad),
                mse]