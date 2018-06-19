import tensorflow as tf
from layers import *

class QANet(object):
    def __init__(self, config, batch, 
        word_mat=None, char_mat=None, 
        trainable=True, opt=True, 
        demo = False, graph = None):

        self.config = config
        self.demo = demo
        self.graph = graph if graph else tf.Graph()

        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)

            self.dropout = tf.placeholder_with_default(0., shape=[], name='dropout')
            if self.demo:
                self.c = tf.placeholder(tf.int32, [None, config.test_para_limit],"context")
                self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit],"question")
                self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit],"context_char")
                self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit],"question_char")
                self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index1")
                self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index2")
            else:
                self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()

            self.word_mat = tf.get_variable('word_mat', 
                initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
            self.char_mat = tf.get_variable('char_mat', 
                initializer=tf.constant(char_mat, dtype=tf.float32), trainable=True)

            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(self.c_mask, axis=-1)
            self.q_len = tf.reduce_sum(self.q_mask, axis=-1)

            if opt:
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])


            self.forward()

            if trainable:
                self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(
                    gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads


        with tf.variable_scope('Input_Embedding_Layer'):
            ch_emb = tf.nn.embedding_lookup(self.char_mat, self.ch)
            qh_emb = tf.nn.embedding_lookup(self.char_mat, self.qh)
            ch_emb = tf.reshape(ch_emb, (N * PL, CL, dc))
            qh_emb = tf.reshape(qh_emb, (N * QL, CL, dc))
            ch_emb = tf.nn.dropout(ch_emb, keep_prob=1 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, keep_prob=1 - 0.5 * self.dropout)

            ch_emb = conv(ch_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = None)
            qh_emb = conv(qh_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = True)

            ch_emb = tf.reduce_max(ch_emb, axis = 1)
            qh_emb = tf.reduce_max(qh_emb, axis = 1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)
            q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)

            






