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
    
    
    
class GHMCLoss:
    """ This class is implemented from https://github.com/xyfZzz/GHM_Loss_Tensorflow/blob/master/ghmc_tensorflow.py
    
    I've tested this class and it works fine on my task. 
    There are some problems on my own realization above mainly caused by condition function.
    
    """
    def __init__(self, bins=10, momentum=0.75):
        self.bins = bins
        self.momentum = momentum
        self.edges_left, self.edges_right = self.get_edges(self.bins)  # edges_left: [bins, 1, 1], edges_right: [bins, 1, 1]
        if momentum > 0:
            self.acc_sum = self.get_acc_sum(self.bins) # [bins]

    def get_edges(self, bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left) # [bins]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-6
        edges_right = tf.constant(edges_right) # [bins]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1, 1]
        return edges_left, edges_right

    def get_acc_sum(self, bins):
        acc_sum = [0.0 for _ in range(bins)]
        return tf.Variable(acc_sum, trainable=False)

    def calc(self, input, target, mask=None, is_mask=False):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        mask [batch_num, class_num]
        """
        edges_left, edges_right = self.edges_left, self.edges_right
        mmt = self.momentum
        # gradient length
        self.g = tf.abs(tf.sigmoid(input) - target) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0) # [1, batch_num, class_num]
        g_greater_equal_edges_left = tf.greater_equal(g, edges_left)# [bins, batch_num, class_num]
        g_less_edges_right = tf.less(g, edges_right)# [bins, batch_num, class_num]
        zero_matrix = tf.cast(tf.zeros_like(g_greater_equal_edges_left), dtype=tf.float32) # [bins, batch_num, class_num]
        if is_mask:
            mask_greater_zero = tf.greater(mask, 0)
            inds = tf.cast(tf.logical_and(tf.logical_and(g_greater_equal_edges_left, g_less_edges_right),
                                          mask_greater_zero), dtype=tf.float32)  # [bins, batch_num, class_num]
            tot = tf.maximum(tf.reduce_sum(tf.cast(mask_greater_zero, dtype=tf.float32)), 1.0)
        else:
            inds = tf.cast(tf.logical_and(g_greater_equal_edges_left, g_less_edges_right),
                           dtype=tf.float32)  # [bins, batch_num, class_num]
            input_shape = tf.shape(input)
            tot = tf.maximum(tf.cast(input_shape[0] * input_shape[1], dtype=tf.float32), 1.0)
        num_in_bin = tf.reduce_sum(inds, axis=[1, 2]) # [bins]
        num_in_bin_greater_zero = tf.greater(num_in_bin, 0) # [bins]
        num_valid_bin = tf.reduce_sum(tf.cast(num_in_bin_greater_zero, dtype=tf.float32))

        # num_in_bin = num_in_bin + 1e-12
        if mmt > 0:
            update = tf.assign(self.acc_sum, tf.where(num_in_bin_greater_zero, mmt * self.acc_sum \
                                  + (1 - mmt) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                self.acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(self.acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = acc_sum + zero_matrix # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)
        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = num_in_bin + zero_matrix # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)
        weights = weights / num_valid_bin
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=input)
        loss = tf.reduce_sum(loss * weights) / tot
        return loss

