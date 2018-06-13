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