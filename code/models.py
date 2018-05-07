# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np

class SiameseNN(object):
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, a_embed = self.add_embeddings()
        with tf.variable_scope('siamese') as scope:
            self.q_trans = self.network(q_embed)
            scope.reuse_variables()
            self.a_trans = self.network(a_embed)
        # 损失和精确度
        self.total_loss = self.add_loss_op(self.q_trans, self.a_trans)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)

    # 输入
    def add_placeholders(self):
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

    # word embeddings
    def add_embeddings(self):
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
        # (batch_size * max_len, embed_size)
        max_len = tf.shape(x)[1]
        x = tf.reshape(x, (-1, x.get_shape()[-1]))
        fc1 = self.fc_layer(x, self.config.hidden_size, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, self.config.hidden_size, "fc2")
        ac2 = tf.nn.relu(fc2)
        # (batch_size, max_len, embed_size)
        ac3 = tf.reshape(ac2, (self.batch_size, max_len, ac2.get_shape()[1]))
        # (batch_size, embed_size)
        ac3 = tf.reduce_mean(ac3, axis=1)
        fc3 = self.fc_layer(ac3, self.config.output_size, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    # 损失节点
    def add_loss_op(self, o1, o2):
        # 此处用cos距离
        norm_o1 = tf.nn.l2_normalize(o1, dim=1)
        norm_o2 = tf.nn.l2_normalize(o2, dim=1)
        self.q_a_cosine = tf.reduce_sum(tf.multiply(o1, o2), 1)

        loss = self.contrastive_loss(self.q_a_cosine, self.y)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return total_loss

    def contrastive_loss(self, Ew, y):
        l_1 = self.config.pos_weight * tf.square(1 - Ew)
        l_0 = tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_mean(y * l_1 + (1 - y) * l_0)
        return loss

    # 训练节点
    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            train_op = opt.minimize(loss, self.global_step)
            return train_op


class SiameseCNN(object):
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, a_embed = self.add_embeddings()
        with tf.variable_scope('siamese') as scope:
            self.q_trans = self.network(q_embed, reuse=False)
            scope.reuse_variables()
            self.a_trans = self.network(a_embed, reuse=True)
        # 损失和精确度
        self.total_loss = self.add_loss_op(self.q_trans, self.a_trans)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)

    # 输入
    def add_placeholders(self):
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

    # word embeddings
    def add_embeddings(self):
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

    def network(self, x, reuse=False):
        # (batch_size, conv_size)
        conv1 = self.conv_layer(x, reuse=reuse)
        # (batch_size, hidden_size)
        fc1 = self.fc_layer(conv1, self.config.hidden_size, "fc1")
        ac1 = tf.nn.relu(fc1)
        # (batch_size, output_size)
        fc2 = self.fc_layer(ac1, self.config.output_size, "fc2")
        return fc2

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.0, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def conv_layer(self, h, reuse=False):
        pool = list()
        max_len = h.get_shape()[1]
        h = tf.reshape(h, [-1, max_len, h.get_shape()[2], 1])
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                conv1_W = tf.get_variable('conv_W', shape=[filter_size, self.config.embedding_size, 1, self.config.num_filters], initializer=tf.truncated_normal_initializer(.0, .01))
                conv1_b = tf.get_variable('conv_b', initializer=tf.constant(0.0, shape=[self.config.num_filters]))
                # pooling层的bias,Q和A分开
                pool_b = tf.get_variable('pool_b', initializer=tf.constant(0.0, shape=[self.config.num_filters]))
                # 卷积
                out = tf.nn.relu((tf.nn.conv2d(h, conv1_W, [1,1,1,1], padding='VALID')+conv1_b))
                # 池化
                out = tf.nn.max_pool(out, [1,max_len-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out = tf.nn.tanh(out+pool_b)
                pool.append(out)
                # 加入正则项
                if not reuse:
                    tf.add_to_collection('total_loss', 0.5 * self.config.l2_reg_lambda * tf.nn.l2_loss(conv1_W))

        total_channels = len(self.config.filter_sizes) * self.config.num_filters
        real_pool = tf.reshape(tf.concat(pool, 3), [self.batch_size, total_channels])
        return real_pool

    # 损失节点
    def add_loss_op(self, o1, o2):
        # 此处用cos距离
        norm_o1 = tf.nn.l2_normalize(o1, dim=1)
        norm_o2 = tf.nn.l2_normalize(o2, dim=1)
        self.q_a_cosine = tf.reduce_sum(tf.multiply(o1, o2), 1)

        loss = self.contrastive_loss(self.q_a_cosine, self.y)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return total_loss

    def contrastive_loss(self, Ew, y):
        l_1 = self.config.pos_weight * tf.square(1 - Ew)
        l_0 = tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_mean(y * l_1 + (1 - y) * l_0)
        return loss

    # 训练节点
    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            train_op = opt.minimize(loss, self.global_step)
            return train_op


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

    # 输入
    def add_placeholders(self):
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

    # word embeddings
    def add_embeddings(self):
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
        return fc2

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def rnn_layer(self, h):
        if self.config.cell_type == 'lstm':
            birnn_fw, birnn_bw = self.bi_lstm(self.config.rnn_size, self.config.layer_size, self.config.keep_prob)
        else:
            birnn_fw, birnn_bw = self.bi_gru(self.config.rnn_size, self.config.layer_size, self.config.keep_prob)
        outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(birnn_fw, birnn_bw, h, dtype=tf.float32)
        # (time_step, batch_size, 2*rnn_size) -> (batch_size, 2*rnn_size)
        output_x1 = tf.reduce_mean(outputs_x1, 0)
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

    # 损失节点
    def add_loss_op(self, o1, o2):
        # 此处用cos距离
        norm_o1 = tf.nn.l2_normalize(o1, dim=1)
        norm_o2 = tf.nn.l2_normalize(o2, dim=1)
        self.q_a_cosine = tf.reduce_sum(tf.multiply(o1, o2), 1)

        loss = self.contrastive_loss(self.q_a_cosine, self.y)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return total_loss

    def contrastive_loss(self, Ew, y):
        l_1 = self.config.pos_weight * tf.square(1 - Ew)
        l_0 = tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_mean(y * l_1 + (1 - y) * l_0)
        return loss

    # 训练节点
    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            train_op = opt.minimize(loss, self.global_step)
            return train_op


class QACNN(object):
    """
    pairwise学习模型
    """
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, aplus_embed, aminus_embed = self.add_embeddings()
        # [batch_size, sequence_size, hidden_size, 1]
        self.h_q, self.h_ap, self.h_am = self.add_hl(q_embed, aplus_embed, aminus_embed)
        # [batch_size, total_channels]
        real_pool_q, real_pool_ap, real_pool_am = self.add_model(q_embed, aplus_embed, aminus_embed)
        # [batch_size, 1]
        self.q_ap_cosine, self.q_am_cosine = self.calc_cosine(real_pool_q, real_pool_ap, real_pool_am)
        # 损失和精确度
        self.total_loss, self.loss, self.accu = self.add_loss_op(self.q_ap_cosine, self.q_am_cosine)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)


    # 输入
    def add_placeholders(self):
        # 问题
        self.q = tf.placeholder(tf.int32,
                shape=[None, self.config.max_q_length],
                name='Question')
        # 正向回答
        self.aplus = tf.placeholder(tf.int32,
                shape=[None, self.config.max_a_length],
                name='PosAns')
        # 负向回答
        self.aminus = tf.placeholder(tf.int32,
                shape=[None, self.config.max_a_length],
                name='NegAns')
        # drop_out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]

    # word embeddings
    def add_embeddings(self):
        with tf.variable_scope('embedding'):
            if self.config.embeddings is not None:
                embeddings = tf.Variable(self.config.embeddings, name="embeddings", trainable=False)
            else:
                embeddings = tf.get_variable('embeddings', shape=[self.config.vocab_size, self.config.embedding_size], initializer=tf.uniform_unit_scaling_initializer())
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            aplus_embed = tf.nn.embedding_lookup(embeddings, self.aplus)
            aminus_embed = tf.nn.embedding_lookup(embeddings, self.aminus)
            q_embed = tf.nn.dropout(q_embed, keep_prob=self.keep_prob)
            aplus_embed = tf.nn.dropout(aplus_embed, keep_prob=self.keep_prob)
            aminus_embed = tf.nn.dropout(aminus_embed, keep_prob=self.keep_prob)
            return q_embed, aplus_embed, aminus_embed

    # Hidden Layer
    def add_hl(self, q_embed, aplus_embed, aminus_embed):
        with tf.variable_scope('HL'):
            W = tf.get_variable('weights', shape=[self.config.embedding_size, self.config.hidden_size], initializer=tf.uniform_unit_scaling_initializer())
            b = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[self.config.hidden_size]))
            h_q = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(q_embed, [-1, self.config.embedding_size]), W)+b), [-1, self.config.max_q_length, self.config.hidden_size])
            h_ap = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(aplus_embed, [-1, self.config.embedding_size]), W)+b), [-1, self.config.max_a_length, self.config.hidden_size])
            h_am = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(aminus_embed, [-1, self.config.embedding_size]), W)+b), [-1, self.config.max_a_length, self.config.hidden_size])
            tf.add_to_collection('total_loss', 0.5*self.config.l2_reg_lambda*tf.nn.l2_loss(W))
            return h_q, h_ap, h_am

    # CNN层
    def add_model(self, h_q, h_ap, h_am):
        pool_q = list()
        pool_ap = list()
        pool_am = list()
        h_q = tf.reshape(h_q, [-1, self.config.max_q_length, self.config.embedding_size, 1])
        h_ap = tf.reshape(h_ap, [-1, self.config.max_a_length, self.config.embedding_size, 1])
        h_am = tf.reshape(h_am, [-1, self.config.max_a_length, self.config.embedding_size, 1])
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                conv1_W = tf.get_variable('W_q', shape=[filter_size, self.config.embedding_size, 1, self.config.num_filters], initializer=tf.truncated_normal_initializer(.0, .1))
                conv2_W = tf.get_variable('W_a', shape=[filter_size, self.config.embedding_size, 1, self.config.num_filters], initializer=tf.truncated_normal_initializer(.0, .1))
                conv1_b = tf.get_variable('conv_qb', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                conv2_b = tf.get_variable('conv_ab', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                # pooling层的bias,Q和A分开
                pool_qb = tf.get_variable('pool_qb', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                pool_ab = tf.get_variable('pool_ab', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                # 卷积
                out_q = tf.nn.relu((tf.nn.conv2d(h_q, conv1_W, [1,1,1,1], padding='VALID')+conv1_b))
                # 池化
                out_q = tf.nn.max_pool(out_q, [1,self.config.max_q_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_q = tf.nn.tanh(out_q+pool_qb)
                pool_q.append(out_q)

                out_ap = tf.nn.relu((tf.nn.conv2d(h_ap, conv2_W, [1,1,1,1], padding='VALID')+conv2_b))
                out_ap = tf.nn.max_pool(out_ap, [1,self.config.max_a_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_ap = tf.nn.tanh(out_ap+pool_ab)
                pool_ap.append(out_ap)

                out_am = tf.nn.relu((tf.nn.conv2d(h_am, conv2_W, [1,1,1,1], padding='VALID')+conv2_b))
                out_am = tf.nn.max_pool(out_am, [1,self.config.max_a_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_am = tf.nn.tanh(out_am+pool_ab)
                pool_am.append(out_am)

                # 加入正则项
                tf.add_to_collection('total_loss', 0.5*self.config.l2_reg_lambda*tf.nn.l2_loss(conv1_W))
                tf.add_to_collection('total_loss', 0.5*self.config.l2_reg_lambda*tf.nn.l2_loss(conv2_W))

        total_channels = len(self.config.filter_sizes)*self.config.num_filters
        real_pool_q = tf.reshape(tf.concat(pool_q, 3), [-1, total_channels])
        real_pool_ap = tf.reshape(tf.concat(pool_ap, 3), [-1, total_channels])
        real_pool_am = tf.reshape(tf.concat(pool_am, 3), [-1, total_channels])

        return real_pool_q, real_pool_ap, real_pool_am

    # 计算cosine
    def calc_cosine(self, real_pool_q, real_pool_ap, real_pool_am):
        normalized_q_h_pool = tf.nn.l2_normalize(real_pool_q, dim=1)
        normalized_pos_h_pool = tf.nn.l2_normalize(real_pool_ap, dim=1)
        normalized_neg_h_pool = tf.nn.l2_normalize(real_pool_am, dim=1)
        q_ap_cosine = tf.reduce_sum(tf.multiply(normalized_q_h_pool, normalized_pos_h_pool), 1)
        q_am_cosine = tf.reduce_sum(tf.multiply(normalized_q_h_pool, normalized_neg_h_pool), 1)

        return q_ap_cosine, q_am_cosine

    # 损失节点
    def add_loss_op(self, q_ap_cosine, q_am_cosine):
        original_loss = self.config.m - q_ap_cosine + q_am_cosine
        l = tf.maximum(tf.zeros_like(original_loss), original_loss)
        loss = tf.reduce_sum(l)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        accu = tf.reduce_mean(tf.cast(tf.equal(0., l), tf.float32))
        return total_loss, loss, accu

    # 训练节点
    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            train_op = opt.minimize(loss, self.global_step)
            return train_op
