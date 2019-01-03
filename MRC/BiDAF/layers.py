# -*- encoding=utf8 -*-

import tensorflow as tf

INF = 1e30

def dropout(inputs, keep_prob, is_train):
    return tf.cond(is_train, lambda: tf.nn.dropout(
            inputs, keep_prob), lambda: inputs)


def softmax_mask(inputs, mask):
    """ Mask the padding values which may affect the softmax calculation.
    inputs: any shape
    mask: the same shape as `inputs`

    """
    return -INF * (1. - tf.cast(mask, tf.float32)) + inputs


def dense(inputs, hidden_size, use_bias=True, scope="dense"):
    """
    inputs: [batch_size, ..., dim]

    return: [batch_size, ..., hidden]
    """
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        last_dim = inputs.get_shape().as_list()[-1]
        output_shape = [shape[_] for _ in range(len(inputs.get_shape().as_list()) - 1)] + [hidden_size]
        flat_inp = tf.reshape(inputs, (-1, last_dim))
        W = tf.get_variable('W', shape=[last_dim, hidden_size], dtype=tf.float32)
        out = tf.matmul(flat_inp, W)
        if use_bias:
            b = tf.get_variable('b', shape=[hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            out = tf.nn.bias_add(out, b)
        out = tf.reshape(out, output_shape)
        return out


def pointer(inputs, state, hidden_size, mask, scope="pointer"):
    """
    inputs: [batch_size, seq_len, dim1]
    state: [batch_size, dim2]
    mask: [batch_size, seq_len]

    return: [batch_size, hidden_size], [batch_size, seq_len]
    """
    with tf.variable_scope(scope):
        u = tf.concat(
            [inputs, tf.tile(
                tf.expand_dims(state, axis=1), [1, tf.shape(inputs)[1], 1])], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden_size, False, scope='s0'))
        s = dense(s0, 1, False, scope='s')
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(inputs * a, axis=1)
        return res, s1

def summ(memory, hidden_size, mask, keep_prob=1.0, is_train=None, scope="summ"):
    """
    memory: [batch_size, seq_len, dim1]

    return: [batch_size, hidden_size]
    """
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden_size, scope='s0'))
        s = dense(s0, 1, False, 's')
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(memory * a, axis=1)
        return res


class cudnn_gru(object):
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope='cudnn_gru'):
        self.scope = scope
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.masks = []
        self.params = []
        for layer in range(self.num_layers):
            input_size_ = input_size if layer == 0 else num_units * 2
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, input_size_)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units, input_size_)
            self.grus.append((gru_fw, gru_bw))

            param_fw = tf.Variable(tf.random_uniform(
                [gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
            param_bw = tf.Variable(tf.random_uniform(
                [gru_bw.params_size()], -0.1, 0.1), validate_shape=False)
            self.params.append((param_fw, param_bw))

            init_fw = tf.Variable(tf.zeros([1, batch_size, num_units]), trainable=False)
            init_bw = tf.Variable(tf.zeros([1, batch_size, num_units]), trainable=False)
            # init_fw = tf.tile(tf.zeros((1, 1, num_units), dtype=tf.float32), (1, batch_size, 1))
            # init_bw = tf.tile(tf.zeros((1, 1, num_units), dtype=tf.float32), (1, batch_size, 1))
            self.inits.append((init_fw, init_bw))

            mask_fw = dropout(tf.Variable(tf.ones((1, batch_size, input_size_), dtype=tf.float32), trainable=False), keep_prob=keep_prob, is_train=is_train)
            mask_bw = dropout(tf.Variable(tf.ones((1, batch_size, input_size_), dtype=tf.float32), trainable=False), keep_prob=keep_prob, is_train=is_train)
            self.masks.append((mask_fw, mask_bw))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat=True):
        """
        inputs: [batch_size, seq_len, dim]

        return: [batch_size, seq_len, num_units * 2 * n] or [batch_size, seq_len, num_units * 2]
        """
        outputs = [tf.transpose(inputs, (1, 0, 2))]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.masks[layer]
                param_fw, param_bw = self.params[layer]
                with tf.variable_scope('fw_{}'.format(layer)):
                    # out_fw, _ = gru_fw(outputs[-1] * mask_fw, initial_state=(init_fw, ))
                    out_fw, _ = gru_fw(outputs[-1] * mask_fw, init_fw, param_fw)
                with tf.variable_scope('bw_{}'.format(layer)):
                    input_bw = tf.reverse_sequence(outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    # out_bw, _ = gru_bw(input_bw, initial_state=(init_bw, ))
                    out_bw, _ = gru_bw(input_bw, init_bw, param_bw)
                    out_bw = tf.reverse_sequence(out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)

                outputs.append(tf.concat((out_fw, out_bw), axis=2))

            if concat:
                res = tf.concat(outputs, axis=2)
            else:
                res = outputs[-1]
            res = tf.transpose(res, (1, 0, 2))
            return res

class native_gru(object):
    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope='native_gru'):
        self.scope = scope
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.masks = []
        for layer in range(self.num_layers):
            input_size_ = input_size if layer == 0 else num_units * 2
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            self.grus.append((gru_fw, gru_bw))
            init_fw = tf.tile(tf.zeros((1, num_units), dtype=tf.float32), (batch_size, 1))
            init_bw = tf.tile(tf.zeros((1, num_units), dtype=tf.float32), (batch_size, 1))
            self.inits.append((init_fw, init_bw))
            mask_fw = dropout(tf.ones((batch_size, 1, input_size_), dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)
            mask_bw = dropout(tf.ones((batch_size, 1, input_size_), dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)
            self.masks.append((mask_fw, mask_bw))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat=True):
        """
        inputs: [batch_size, seq_len, dim]
        seq_len: [batch_size, ]

        return: [batch_size, seq_len, num_units * 2 * n] or [batch_size, seq_len, num_units * 2]
        """
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.masks[layer]
                with tf.variable_scope('fw_{}'.format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(gru_fw, outputs[-1] * mask_fw, sequence_length=seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope('bw_{}'.format(layer)):
                    input_bw = tf.reverse_sequence(outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(gru_bw, input_bw, sequence_length=seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)

                outputs.append(tf.concat((out_fw, out_bw), axis=2))

            if concat:
                res = tf.concat(outputs[1:], axis=2)
            else:
                res = outputs[-1]
            return res



class ptr_layer(object):
    def __init__(self, batch_size, hidden_size, is_train=None, keep_prob=1.0, scope='ptr_net'):
        self.gru = tf.contrib.rnn.GRUCell(hidden_size)
        self.scope = scope
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.is_train = is_train
        self.keep_prob = keep_prob
        self.dp_mask = dropout(tf.ones([batch_size, hidden_size]), keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask):
        """
        init: [batch_size, hidden_size]
        match: [batch_size, seq_len, dim]
        mask: [batch_size, seq_len]

        return: [batch_size, seq_len], [batch_size, seq_len]
        """
        with tf.variable_scope(self.scope):
            d_match = dropout(match, keep_prob=self.keep_prob, is_train=self.is_train)
            inp, logits1 = pointer(d_match, init * self.dp_mask, d, mask)
            d_inp = dropout(inp, keep_prob=self.keep_prob, is_train=self.is_train)
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(d_match, state * self.dp_mask, d, mask)
            return logits1, logits2

    

