"""
The implementation of GHM-C and GHM-R losses.
Details can be found in the paper `Gradient Harmonized Single-stage Detector`:
https://arxiv.org/abs/1811.05181

Reference 
[1] https://github.com/libuyu/GHM_Detection/blob/master/mmdetection/mmdet/core/loss/ghm_loss.py
[2] https://arxiv.org/abs/1811.05181
"""

import tensorflow as tf


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) + (1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


class GHMC_loss(object):
    def __init__(self, bins=10, momentum=0.9):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [tf.Variable(0.0, trainable=False) for _ in range(bins)]
    
    def __call__(self, input, target, mask=None):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.

        return: a scalar loss
        """
        edges = self.edges
        mmt = self.momentum
        weights = tf.zeros_like(input, dtype=tf.float32)
        if mask is None:
            mask = tf.ones_like(input, dtype=tf.float32)

        def func1(weights, n):
            if mmt > 0:
                tf.assign(self.acc_sum[i], mmt * self.acc_sum[i] \
                    + (1 - mmt) * num_in_bin)
                weights = weights + weights * tf.cast(inds, tf.float32) * (tot / self.acc_sum[i])
            else:
                weights = weights + weights * tf.cast(inds, tf.float32) * (tot / num_in_bin)
            n += 1
            return (weights, n)

        # gradient length
        g = tf.abs(tf.nn.sigmoid(input) - target)

        valid = mask > 0
        tot = tf.maximum(tf.reduce_sum(tf.cast(valid, tf.float32)), 1.0)
        n = tf.Variable(0, trainable=False, dtype=tf.float32)  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = tf.reduce_sum(tf.cast(inds, tf.float32))
            weights, n = tf.cond(num_in_bin > 0, lambda: func1(weights, n), lambda: (weights, n))
        weights = tf.cond(n > 0, lambda: weights / n, lambda: weights)

        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=input * weights, labels=target)) / tot
        return loss

class GHMR_loss(object):
    def __init__(self, mu=0.02, bins=10, momentum=0.9):
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [tf.Variable(0.0, trainable=False) for _ in range(bins)]


    def __call__(self, input, target, mask=None):
        """ Args:
        input [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = input - target
        loss = tf.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = tf.abs(diff / tf.sqrt(mu * mu + diff * diff))
        weights = tf.zeros_like(g, dtype=tf.float32)
        if mask is None:
            mask = tf.ones_like(input, dtype=tf.float32)

        def func1(weights, n):
            if mmt > 0:
                tf.assign(self.acc_sum[i], mmt * self.acc_sum[i] \
                    + (1 - mmt) * num_in_bin)
                weights = weights + weights * tf.cast(inds, tf.float32) * (tot / self.acc_sum[i])
            else:
                weights = weights + weights * tf.cast(inds, tf.float32) * (tot / num_in_bin)
            n += 1
            return (weights, n)

        valid = mask > 0
        tot = tf.maximum(tf.reduce_sum(tf.cast(valid, tf.float32)), 1.0)
        n = tf.Variable(0, trainable=False, dtype=tf.float32)  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = tf.reduce_sum(tf.cast(inds, tf.float32))
            weights, n = tf.cond(num_in_bin > 0, lambda: func1(weights, n), lambda: (weights, n))
        weights = tf.cond(n > 0, lambda: weights / n, lambda: weights)

        loss = loss * weights
        loss = tf.reduce_sum(loss) / tot
        return loss

