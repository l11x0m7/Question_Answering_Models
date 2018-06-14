# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np


class SeqMatchSeq(object):
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, a_embed = self.add_embeddings()
        # 上下文编码
        q_encode, a_encode = self.context_encoding(q_embed, a_embed)
        # attention层
        h_a = self.attend(q_encode, a_encode)
        # compose层
        t = self.compare(a_encode, h_a)
        # aggregate层
        agg_out = self.aggregate(t)
        pred = self.soft_out(agg_out)
        # 预测概率分布与损失
        self.y_hat, self.total_loss = self.add_loss_op(pred)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)

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
                embeddings = tf.Variable(self.config.embeddings, 
                    name="embeddings", trainable=False)
            else:
                embeddings = tf.get_variable('embeddings', 
                    shape=[self.config.vocab_size, self.config.embedding_size], 
                    initializer=tf.uniform_unit_scaling_initializer())
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            a_embed = tf.nn.embedding_lookup(embeddings, self.a)
            return q_embed, a_embed

    def context_encoding(self, q, a):
        """
        q: [batch_size, q_length, embedding_dim]
        a: [batch_size, a_length, embedding_dim]
        """
        with tf.variable_scope('context_encoding') as scope:
            q_encode = self.proj_layer(q, 'proj_layer', reuse=None)
            a_encode = self.proj_layer(a, 'proj_layer', reuse=True)
        return q_encode, a_encode


    def attend(self, q, a):
        """
        q: [batch_size, q_length, represent_dim]
        a: [batch_size, a_length, represent_dim]
        """
        q_proj = self.mlp(q, self.config.mem_dim, 1, None, 
                    'att_q_proj', reuse=None)
        # [batch_size, q_length, a_length]
        att_inner_product = tf.matmul(q_proj, tf.transpose(a, (0, 2, 1)))
        # [batch_size, a_length, q_length]
        q_weights = tf.nn.softmax(
                        tf.transpose(
                            att_inner_product, (0, 2, 1)), dim=-1)
        output_a = tf.matmul(q_weights, q)
        return output_a

    def compare(self, a, h_a):
        """
        a: [batch_size, a_length, mem_dim]
        a_att: [batch_size, a_length, mem_dim]
        """
        if self.config.comp_type == 'mul':
            out = a * h_a
        else:
            raise ValueError('{} method is not implemented!'.format(
                self.config.comp_type))

        return out

    def aggregate(self, t):
        """
        t: [batch_size, a_length, mem_dim]
        """
        pool_t = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                # 卷积
                out_t = tf.layers.Conv1D(self.config.cov_dim,
                                         filter_size,
                                         strides=1,
                                         padding='valid',
                                         activation=tf.nn.relu, name='conv')(t)
                # 池化
                out_t = tf.layers.MaxPooling1D(
                    self.config.max_a_length - filter_size + 1, 
                    1, name='max_pool')(out_t)
                out_t = tf.reshape(out_t, 
                    (tf.shape(out_t)[0], out_t.get_shape().as_list()[2]))
                pool_t.append(out_t)
        # [batch_size, n * mem_dim]
        out = tf.concat(pool_t, axis=-1)
        # [batch_size, mem_dim]
        out = self.mlp(out, self.config.mem_dim, 1, 
                        tf.nn.tanh, 'pre_out', use_dropout=False, reuse=None)
        return out

    def soft_out(self, x):
        out = self.mlp(x, 2, 1, None, 
            'soft_out', use_dropout=False, reuse=None)
        return out

    def mlp(self, bottom, size, layer_num, activation, name, use_dropout=True, reuse=None):
        """
        bottom: 上层输入
        size: 神经元大小
        layer_num: 神经网络层数
        name: mlp的名称
        reuse: 是否复用层
        """
        now = bottom
        if use_dropout:
            now = tf.nn.dropout(now, keep_prob=self.keep_prob)
        for i in xrange(layer_num):
            now = tf.layers.dense(now, size, 
                                  activation=activation, 
                                  name=name + '_{}'.format(i), 
                                  reuse=reuse)
        return now

    def proj_layer(self, seq, name, reuse=None):
        out1 = self.mlp(seq, self.config.mem_dim, 1, 
                    tf.nn.sigmoid, name + '_sigmoid', reuse=reuse)
        out2 = self.mlp(seq, self.config.mem_dim, 1, 
                    tf.nn.tanh, name + '_tanh', reuse=reuse)
        out = out1 * out2
        return out

    def add_loss_op(self, pred):
        """
        损失节点
        """
        # [batch_size, 2]
        y_hat = tf.nn.softmax(pred, dim=-1)
        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(self.y, pred))
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return y_hat, total_loss

    def add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, 
                    name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            # train_op = opt.minimize(loss, self.global_step)
            train_variables = tf.trainable_variables()
            grads_vars = opt.compute_gradients(loss, train_variables)
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (
                    tf.clip_by_norm(grad, self.config.grad_clip), var)
            train_op = opt.apply_gradients(
                grads_vars, global_step=self.global_step)
            return train_op
