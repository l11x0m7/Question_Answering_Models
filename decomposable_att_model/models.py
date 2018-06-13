# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np


class DecompAtt(object):
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, a_embed = self.add_embeddings()
        # 上下文编码
        q_encode, a_encode = self.context_encoding(q_embed, a_embed)
        # attention层
        q_attend, a_attend = self.attend(q_encode, a_encode)
        # compose层
        q_comp, a_comp = self.compare(q_encode, a_encode, q_attend, a_attend)
        # aggregate层
        pred = self.aggregate(q_comp, a_comp)
        # 预测概率分布与损失
        self.y_hat, self.total_loss = self.add_loss_op(pred)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)


    def attend(self, q, a):
        """
        q: [batch_size, q_length, represent_dim]
        a: [batch_size, a_length, represent_dim]
        """

        q = tf.nn.dropout(q, keep_prob=self.keep_prob)
        a = tf.nn.dropout(a, keep_prob=self.keep_prob)
        q_map = tf.layers.dense(q, 128, activation=tf.nn.relu, name='embed_map')
        a_map = tf.layers.dense(a, 128, activation=tf.nn.relu, name='embed_map', reuse=True)
        # [batch_size, q_length, a_length]
        att_inner_product = tf.matmul(
            q_map,
            tf.transpose(a_map, [0, 2, 1]))
        # [batch_size, a_length, q_length]
        q_weights = tf.nn.softmax(
                        tf.transpose(
                            att_inner_product, (0, 2, 1)), dim=-1)
        # [batch_size, q_length, a_length]
        a_weights = tf.nn.softmax(att_inner_product, dim=-1)

        output_a = tf.matmul(q_weights, q)
        output_q = tf.matmul(a_weights, a)

        return output_q, output_a

    def compare(self, q, a, q_att, a_att):
        """
        q: [batch_size, q_length, represent_dim]
        a: [batch_size, a_length, represent_dim]
        q_att: [batch_size, q_length, represent_dim]
        a_att: [batch_size, a_length, represent_dim]
        """
        q_combine = tf.concat([q, q_att], axis=-1)
        a_combine = tf.concat([a, a_att], axis=-1)
        q_combine = tf.nn.dropout(q_combine, keep_prob=self.keep_prob)
        a_combine = tf.nn.dropout(a_combine, keep_prob=self.keep_prob)
        q_map = self.mlp(q_combine, self.config.hidden_size, 2, 'embed_compare')
        a_map = self.mlp(a_combine, self.config.hidden_size, 2, 'embed_compare', reuse=True)
        return q_map, a_map

    def aggregate(self, q, a):
        """
        q: [batch_size, q_length, represent_dim]
        a: [batch_size, a_length, represent_dim]
        """
        # 输出shape为[batch_size, represent_dim]
        q_sum = tf.reduce_sum(q, 1)
        a_sum = tf.reduce_sum(a, 1)
        q_sum = tf.nn.dropout(q_sum, keep_prob=self.keep_prob)
        a_sum = tf.nn.dropout(a_sum, keep_prob=self.keep_prob)
        q_a_rep = tf.concat([q_sum, a_sum], axis=-1)
        pred = self.mlp(q_a_rep, self.config.output_size, 2, 'embed_aggregate')
        pred = tf.layers.dense(pred, 2, activation=None, name='prediction')
        return pred

    def add_placeholders(self):
        # 问题
        self.q = tf.placeholder(tf.int32,
                shape=[None, self.config.max_q_length],
                name='Question')
        # 回答
        self.a = tf.placeholder(tf.int32,
                shape=[None, self.config.max_a_length],
                name='Ans')
        self.y = tf.placeholder(tf.int32, shape=[None, ], name='label')
        # drop_out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]

    def add_embeddings(self):
        with tf.variable_scope('embedding'):
            if self.config.embeddings is not None:
                embeddings = tf.Variable(self.config.embeddings, name="embeddings", trainable=False)
            else:
                embeddings = tf.get_variable('embeddings', shape=[self.config.vocab_size, self.config.embedding_size], initializer=tf.uniform_unit_scaling_initializer())
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            a_embed = tf.nn.embedding_lookup(embeddings, self.a)
            return q_embed, a_embed

    def context_encoding(self, q, a):
        """
        q: [batch_size, q_length, embedding_dim]
        a: [batch_size, a_length, embedding_dim]
        """
        with tf.variable_scope('context_encoding') as scope:
            q = tf.nn.dropout(q, keep_prob=self.keep_prob)
            a = tf.nn.dropout(a, keep_prob=self.keep_prob)
            q_encode = self.rnn_layer(q)
            tf.get_variable_scope().reuse_variables() 
            a_encode = self.rnn_layer(a)
        return q_encode, a_encode

    def mlp(self, bottom, size, layer_num, name, reuse=None):
        """
        bottom: 上层输入
        size: 神经元大小
        layer_num: 神经网络层数
        name: mlp的名称
        reuse: 是否复用层
        """
        now = bottom
        for i in xrange(layer_num):
            now = tf.layers.dense(now, 128, 
                                  activation=tf.nn.relu, 
                                  name=name + '_{}'.format(i), 
                                  reuse=reuse)
        return now

    def rnn_layer(self, h):
        sequence_length = h.get_shape()[1]
        # (batch_size, time_step, embed_size) -> (time_step, batch_size, embed_size)
        inputs = tf.transpose(h, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, self.config.embedding_size])
        inputs = tf.split(inputs, sequence_length, 0)

        if self.config.cell_type == 'lstm':
            birnn_fw, birnn_bw = self.bi_lstm(self.config.rnn_size, self.config.layer_size, self.config.keep_prob)
        else:
            birnn_fw, birnn_bw = self.bi_gru(self.config.rnn_size, self.config.layer_size, self.config.keep_prob)
        outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(birnn_fw, birnn_bw, inputs, dtype=tf.float32)
        # (time_step, batch_size, 2*rnn_size) -> (batch_size, time_step, 2*rnn_size)
        output_x1 = tf.transpose(outputs_x1, (1, 0, 2))
        return output_x1

    def bi_lstm(self, rnn_size, layer_size, keep_prob):
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
        # forward rnn
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            gru_fw_cell_list = [tf.contrib.rnn.GRUCell(rnn_size) for _ in xrange(layer_size)]
            gru_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(gru_fw_cell_list), output_keep_prob=keep_prob)

        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            gru_bw_cell_list = [tf.contrib.rnn.GRUCell(rnn_size) for _ in xrange(layer_size)]
            gru_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(gru_bw_cell_list), output_keep_prob=keep_prob)

        return gru_fw_cell_m, gru_bw_cell_m

    def add_loss_op(self, pred):
        """
        损失节点
        """
        # [batch_size, 2]
        y_hat = tf.nn.softmax(pred, dim=-1)
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.y, pred))
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return y_hat, total_loss

    def add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            # train_op = opt.minimize(loss, self.global_step)
            train_variables = tf.trainable_variables()
            grads_vars = opt.compute_gradients(loss, train_variables)
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, self.config.grad_clip), var)
            train_op = opt.apply_gradients(grads_vars, global_step=self.global_step)
            return train_op
