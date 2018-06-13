# -*- encoding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np


class BiMPM(object):
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






# 以下代码参考https://github.com/zhiguowang/BiMPM/blob/master/src/layer_utils.py

def my_lstm_layer(input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False, is_training=True,
                  dropout_rate=0.2, use_cudnn=True):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param lstm_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name, reuse=reuse):
        if use_cudnn:
            inputs = tf.transpose(input_reps, [1, 0, 2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, direction="bidirectional",
                                    name="{}_cudnn_bi_lstm".format(scope_name), dropout=dropout_rate if is_training else 0)
            outputs, _ = lstm(inputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            f_rep = outputs[:, :, 0:lstm_dim]
            b_rep = outputs[:, :, lstm_dim:2*lstm_dim]
        else:
            context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            if is_training:
                context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
            context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
            context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

            (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
            outputs = tf.concat(axis=2, values=[f_rep, b_rep])
    return (f_rep,b_rep, outputs)

def dropout_layer(input_reps, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr

def cosine_distance(y1,y2, cosine_norm=True, eps=1e-6):
    # cosine_norm = True
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm

def euclidean_distance(y1, y2, eps=1e-6):
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    return distance

def cross_entropy(logits, truth, mask=None):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]
    if mask is not None: logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
    result = tf.multiply(truth, log_predictions) # [batch_size, passage_len]
    if mask is not None: result = tf.multiply(result, mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]

def projection_layer(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
    with tf.variable_scope(scope or "projection_layer"):
        full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        outputs = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs # [batch_size, passage_len, output_size]

def highway_layer(in_val, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, activation_func=tf.tanh, scope_name=None, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        for i in xrange(num_layers):
            cur_scope_name = scope_name + "-{}".format(i)
            in_val = highway_layer(in_val, output_size,activation_func=activation_func, scope=cur_scope_name)
    return in_val

def collect_representation(representation, positions):
    # representation: [batch_size, node_num, feature_dim]
    # positions: [batch_size, neigh_num]
    return collect_probs(representation, positions)

def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
    return result # [batch_size, dim]

def collect_probs(probs, positions):
    # probs [batch_size, chunks_size]
    # positions [batch_size, pair_size]
    batch_size = tf.shape(probs)[0]
    pair_size = tf.shape(positions)[1]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    batch_nums = tf.reshape(batch_nums, shape=[-1, 1]) # [batch_size, 1]
    batch_nums = tf.tile(batch_nums, multiples=[1, pair_size]) # [batch_size, pair_size]

    indices = tf.stack((batch_nums, positions), axis=2) # shape (batch_size, pair_size, 2)
    pair_probs = tf.gather_nd(probs, indices)
    # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
    return pair_probs


def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None, is_training=False, dropout_rate=0.2):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])


        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * diagnoal_params
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]

        # normalize
        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value

def weighted_sum(atten_scores, in_values):
    '''
    :param atten_scores: # [batch_size, len1, len2]
    :param in_values: [batch_size, len2, dim]
    :return:
    '''
    return tf.matmul(atten_scores, in_values)

def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    if question_mask is not None:
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]





# 以下代码参考https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py

eps = 1e-6
def cosine_distance(y1,y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps)) 
    return cosine_numerator / y1_norm / y2_norm

def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def multi_perspective_expand_for_3D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=2) #[batch_size, passage_len, 'x', dim]
    decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0), axis=0) # [1, 1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)#[batch_size, passage_len, decompse_dim, dim]

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=1) #[batch_size, 'x', dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0) # [1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params) # [batch_size, decompse_dim, dim]


def cal_maxpooling_matching(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [question_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1) # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, question_len, decompose_dim, dim]
        return cosine_distance(p, q) # [passage_len, question_len, decompose]
    elems = (passage_rep, question_rep)
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat(axis=2, values=[tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])# [batch_size, passage_len, 2*decompse_dim]

def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

#     xdev = x - x.max()
#     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)
    xdev = tf.sub(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.sub(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
#     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions), mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]
    
def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans * gate + in_val * (1.0 - gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in xrange(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val

def cal_max_question_representation(question_representation, atten_scores):
    atten_positions = tf.argmax(atten_scores, axis=2, output_type=tf.int32)  # [batch_size, passage_len]
    max_question_reps = layer_utils.collect_representation(question_representation, atten_positions)
    return max_question_reps

def multi_perspective_match(feature_dim, repres1, repres2, is_training=True, dropout_rate=0.2,
                            options=None, scope_name='mp-match', reuse=False):
    '''
        :param repres1: [batch_size, len, feature_dim]
        :param repres2: [batch_size, len, feature_dim]
        :return:
    '''
    input_shape = tf.shape(repres1)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    matching_result = []
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0
        if options.with_cosine:
            cosine_value = layer_utils.cosine_distance(repres1, repres2, cosine_norm=False)
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            matching_result.append(cosine_value)
            match_dim += 1

        if options.with_mp_cosine:
            mp_cosine_params = tf.get_variable("mp_cosine", shape=[options.cosine_MP_dim, feature_dim], dtype=tf.float32)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            repres1_flat = tf.expand_dims(repres1, axis=2)
            repres2_flat = tf.expand_dims(repres2, axis=2)
            mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(repres1_flat, mp_cosine_params),
                                                             repres2_flat,cosine_norm=False)
            matching_result.append(mp_cosine_matching)
            match_dim += options.cosine_MP_dim

    matching_result = tf.concat(axis=2, values=matching_result)
    return (matching_result, match_dim)


def match_passage_with_question(passage_reps, question_reps, passage_mask, question_mask, passage_lengths, question_lengths,
                                context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                                is_training=True, options=None, dropout_rate=0, forward=True):
    passage_reps = tf.multiply(passage_reps, tf.expand_dims(passage_mask,-1))
    question_reps = tf.multiply(question_reps, tf.expand_dims(question_mask,-1))
    all_question_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        relevancy_matrix = cal_relevancy_matrix(question_reps, passage_reps)
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask)
        # relevancy_matrix = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
        #             scope_name="fw_attention", att_type=options.att_type, att_dim=options.att_dim,
        #             remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)

        all_question_aware_representatins.append(tf.reduce_max(relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(relevancy_matrix, axis=2,keep_dims=True))
        dim += 2
        if with_full_match:
            if forward:
                question_full_rep = layer_utils.collect_final_step_of_lstm(question_reps, question_lengths - 1)
            else:
                question_full_rep = question_reps[:,0,:]

            passage_len = tf.shape(passage_reps)[1]
            question_full_rep = tf.expand_dims(question_full_rep, axis=1)
            question_full_rep = tf.tile(question_full_rep, [1, passage_len, 1])  # [batch_size, pasasge_len, feature_dim]

            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                                passage_reps, question_full_rep, is_training=is_training, dropout_rate=options.dropout_rate,
                                options=options, scope_name='mp-match-full-match')
            all_question_aware_representatins.append(attentive_rep)
            dim += match_dim

        if with_maxpool_match:
            maxpooling_decomp_params = tf.get_variable("maxpooling_matching_decomp",
                                                          shape=[options.cosine_MP_dim, context_lstm_dim], dtype=tf.float32)
            maxpooling_rep = cal_maxpooling_matching(passage_reps, question_reps, maxpooling_decomp_params)
            all_question_aware_representatins.append(maxpooling_rep)
            dim += 2*options.cosine_MP_dim

        if with_attentive_match:
            atten_scores = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
                    scope_name="attention", att_type=options.att_type, att_dim=options.att_dim,
                    remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)
            att_question_contexts = tf.matmul(atten_scores, question_reps)
            (attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                    passage_reps, att_question_contexts, is_training=is_training, dropout_rate=options.dropout_rate,
                    options=options, scope_name='mp-match-att_question')
            all_question_aware_representatins.append(attentive_rep)
            dim += match_dim

        if with_max_attentive_match:
            max_att = cal_max_question_representation(question_reps, relevancy_matrix)
            (max_attentive_rep, match_dim) = multi_perspective_match(context_lstm_dim,
                    passage_reps, max_att, is_training=is_training, dropout_rate=options.dropout_rate,
                    options=options, scope_name='mp-match-max-att')
            all_question_aware_representatins.append(max_attentive_rep)
            dim += match_dim

        all_question_aware_representatins = tf.concat(axis=2, values=all_question_aware_representatins)
    return (all_question_aware_representatins, dim)
    
def bilateral_match_func(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, passage_mask, input_dim, is_training, options=None):

    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    # ====word level matching======
    (match_reps, match_dim) = match_passage_with_question(in_passage_repres, in_question_repres, passage_mask, question_mask, passage_lengths,
                                question_lengths, input_dim, scope="word_match_forward",
                                with_full_match=False, with_maxpool_match=options.with_maxpool_match,
                                with_attentive_match=options.with_attentive_match,
                                with_max_attentive_match=options.with_max_attentive_match,
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=True)
    question_aware_representatins.append(match_reps)
    question_aware_dim += match_dim

    (match_reps, match_dim) = match_passage_with_question(in_question_repres, in_passage_repres, question_mask, passage_mask, question_lengths,
                                passage_lengths, input_dim, scope="word_match_backward",
                                with_full_match=False, with_maxpool_match=options.with_maxpool_match,
                                with_attentive_match=options.with_attentive_match,
                                with_max_attentive_match=options.with_max_attentive_match,
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=False)
    passage_aware_representatins.append(match_reps)
    passage_aware_dim += match_dim

    with tf.variable_scope('context_MP_matching'):
        for i in xrange(options.context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                # contextual lstm for both passage and question
                in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
                in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(passage_mask, axis=-1))
                (question_context_representation_fw, question_context_representation_bw,
                 in_question_repres) = layer_utils.my_lstm_layer(
                        in_question_repres, options.context_lstm_dim, input_lengths= question_lengths,scope_name="context_represent",
                        reuse=False, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
                (passage_context_representation_fw, passage_context_representation_bw, 
                 in_passage_repres) = layer_utils.my_lstm_layer(
                        in_passage_repres, options.context_lstm_dim, input_lengths=passage_lengths, scope_name="context_represent",
                        reuse=True, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)

                # Multi-perspective matching
                with tf.variable_scope('left_MP_matching'):
                    (match_reps, match_dim) = match_passage_with_question(passage_context_representation_fw,
                                question_context_representation_fw, passage_mask, question_mask, passage_lengths,
                                question_lengths, options.context_lstm_dim, scope="forward_match",
                                with_full_match=options.with_full_match, with_maxpool_match=options.with_maxpool_match,
                                with_attentive_match=options.with_attentive_match,
                                with_max_attentive_match=options.with_max_attentive_match,
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=True)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim
                    (match_reps, match_dim) = match_passage_with_question(passage_context_representation_bw,
                                question_context_representation_bw, passage_mask, question_mask, passage_lengths,
                                question_lengths, options.context_lstm_dim, scope="backward_match",
                                with_full_match=options.with_full_match, with_maxpool_match=options.with_maxpool_match,
                                with_attentive_match=options.with_attentive_match,
                                with_max_attentive_match=options.with_max_attentive_match,
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=False)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim

                with tf.variable_scope('right_MP_matching'):
                    (match_reps, match_dim) = match_passage_with_question(question_context_representation_fw,
                                passage_context_representation_fw, question_mask, passage_mask, question_lengths,
                                passage_lengths, options.context_lstm_dim, scope="forward_match",
                                with_full_match=options.with_full_match, with_maxpool_match=options.with_maxpool_match,
                                with_attentive_match=options.with_attentive_match,
                                with_max_attentive_match=options.with_max_attentive_match,
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=True)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim
                    (match_reps, match_dim) = match_passage_with_question(question_context_representation_bw,
                                passage_context_representation_bw, question_mask, passage_mask, question_lengths,
                                passage_lengths, options.context_lstm_dim, scope="backward_match",
                                with_full_match=options.with_full_match, with_maxpool_match=options.with_maxpool_match,
                                with_attentive_match=options.with_attentive_match,
                                with_max_attentive_match=options.with_max_attentive_match,
                                is_training=is_training, options=options, dropout_rate=options.dropout_rate, forward=False)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim

    question_aware_representatins = tf.concat(axis=2, values=question_aware_representatins) # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat(axis=2, values=passage_aware_representatins) # [batch_size, question_len, question_aware_dim]

    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - options.dropout_rate))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - options.dropout_rate))
        
    # ======Highway layer======
    if options.with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(question_aware_representatins, question_aware_dim,
                                                                options.highway_layer_num)
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(passage_aware_representatins, passage_aware_dim,
                                                               options.highway_layer_num)

    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    
    qa_aggregation_input = question_aware_representatins
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in xrange(options.aggregation_layer_num): # support multiple aggregation layer
            qa_aggregation_input = tf.multiply(qa_aggregation_input, tf.expand_dims(passage_mask, axis=-1))
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                        qa_aggregation_input, options.aggregation_lstm_dim, input_lengths=passage_lengths, scope_name='left_layer-{}'.format(i),
                        reuse=False, is_training=is_training, dropout_rate=options.dropout_rate,use_cudnn=options.use_cudnn)
            fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, passage_lengths - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2* options.aggregation_lstm_dim
            qa_aggregation_input = cur_aggregation_representation# [batch_size, passage_len, 2*aggregation_lstm_dim]

            pa_aggregation_input = tf.multiply(pa_aggregation_input, tf.expand_dims(question_mask, axis=-1))
            (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                        pa_aggregation_input, options.aggregation_lstm_dim,
                        input_lengths=question_lengths, scope_name='right_layer-{}'.format(i),
                        reuse=False, is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, question_lengths - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2* options.aggregation_lstm_dim
            pa_aggregation_input = cur_aggregation_representation# [batch_size, passage_len, 2*aggregation_lstm_dim]

    aggregation_representation = tf.concat(axis=1, values=aggregation_representation) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if options.with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, options.highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
    
    return (aggregation_representation, aggregation_dim)

