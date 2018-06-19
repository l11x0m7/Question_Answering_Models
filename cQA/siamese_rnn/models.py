# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np


class SiameseRNN(object):
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, a_embed = self.add_embeddings()
        with tf.variable_scope('siamese') as scope:
            self.q_trans = self.network(q_embed)
            tf.get_variable_scope().reuse_variables()
            self.a_trans = self.network(a_embed)
        # 损失和精确度
        self.total_loss = self.add_loss_op(self.q_trans, self.a_trans)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)

    def add_placeholders(self):
        """
        输入的容器
        """
        # 问题
        self.q = tf.placeholder(tf.int32,
                shape=[None, self.config.max_q_length],
                name='Question')
        # 回答
        self.a = tf.placeholder(tf.int32,
                shape=[None, self.config.max_a_length],
                name='Ans')
        self.y = tf.placeholder(tf.float32, shape=[None, ], name='label')
        # drop_out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]

    def add_embeddings(self):
        """
        embedding层
        """
        with tf.variable_scope('embedding'):
            if self.config.embeddings is not None:
                embeddings = tf.Variable(self.config.embeddings, name="embeddings", trainable=False)
            else:
                embeddings = tf.get_variable('embeddings', shape=[self.config.vocab_size, self.config.embedding_size], initializer=tf.uniform_unit_scaling_initializer())
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            a_embed = tf.nn.embedding_lookup(embeddings, self.a)
            q_embed = tf.nn.dropout(q_embed, keep_prob=self.keep_prob)
            a_embed = tf.nn.dropout(a_embed, keep_prob=self.keep_prob)
            return q_embed, a_embed

    def network(self, x):
        """
        核心网络
        """
        sequence_length = x.get_shape()[1]
        # (batch_size, time_step, embed_size) -> (time_step, batch_size, embed_size)
        inputs = tf.transpose(x, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, self.config.embedding_size])
        inputs = tf.split(inputs, sequence_length, 0)
        # (batch_size, rnn_output_size)
        rnn1 = self.rnn_layer(inputs)
        # (batch_size, hidden_size)
        fc1 = self.fc_layer(rnn1, self.config.hidden_size, "fc1")
        ac1 = tf.nn.relu(fc1)
        # (batch_size, output_size)
        fc2 = self.fc_layer(ac1, self.config.output_size, "fc2")
        ac2 = tf.nn.relu(fc2)
        return ac2

    def fc_layer(self, bottom, n_weight, name):
        """
        全连接层
        """
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def rnn_layer(self, h):
        """
        RNN层
        """
        if self.config.cell_type == 'lstm':
            birnn_fw, birnn_bw = self.bi_lstm(self.config.rnn_size, self.config.layer_size, self.config.keep_prob)
        else:
            birnn_fw, birnn_bw = self.bi_gru(self.config.rnn_size, self.config.layer_size, self.config.keep_prob)
        outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(birnn_fw, birnn_bw, h, dtype=tf.float32)
        # (time_step, batch_size, 2*rnn_size) -> (batch_size, 2*rnn_size)
        output_x1 = tf.reduce_mean(outputs_x1, 0)
        return output_x1

    def bi_lstm(self, rnn_size, layer_size, keep_prob):
        """
        双向LSTM
        """
        # forward rnn
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in xrange(layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=keep_prob)

        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in xrange(layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=keep_prob)

        return lstm_fw_cell_m, lstm_bw_cell_m

    def bi_gru(self, rnn_size, layer_size, keep_prob):
        """
        双向GRU
        """
        # forward rnn
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            gru_fw_cell_list = [tf.contrib.rnn.GRUCell(rnn_size) for _ in xrange(layer_size)]
            gru_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(gru_fw_cell_list), output_keep_prob=keep_prob)

        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            gru_bw_cell_list = [tf.contrib.rnn.GRUCell(rnn_size) for _ in xrange(layer_size)]
            gru_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(gru_bw_cell_list), output_keep_prob=keep_prob)

        return gru_fw_cell_m, gru_bw_cell_m

    def add_loss_op(self, o1, o2):
        """
        损失节点
        """
        # 此处用cos距离
        norm_o1 = tf.nn.l2_normalize(o1, dim=1)
        norm_o2 = tf.nn.l2_normalize(o2, dim=1)
        self.q_a_cosine = tf.reduce_sum(tf.multiply(o1, o2), 1)

        loss = self.contrastive_loss(self.q_a_cosine, self.y)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return total_loss

    def contrastive_loss(self, Ew, y):
        """
        contrasive_loss
        """
        l_1 = self.config.pos_weight * tf.square(1 - Ew)
        l_0 = tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_mean(y * l_1 + (1 - y) * l_0)
        return loss

    def add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            train_op = opt.minimize(loss, self.global_step)
            return train_op
