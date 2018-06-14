# -*- encoding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np
import layer_utils

eps = 1e-6

class BiMPM(object):
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, a_embed = self.add_embeddings()
        input_shape = tf.shape(self.q)
        batch_size = input_shape[0]
        question_len = input_shape[1]
        input_shape = tf.shape(self.a)
        passage_len = input_shape[1]
        # 上下文编码
        q_encode, a_encode = self.context_encoding(q_embed, a_embed)
        input_dim = q_encode.get_shape().as_list()[-1]

        mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]

        # ======Highway layer======
        if self.config.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = self.multi_highway_layer(q_encode, input_dim, self.config.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = self.multi_highway_layer(a_encode, input_dim, self.config.highway_layer_num)

        # in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
        # in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(mask, axis=-1))

        # ========Bilateral Matching=====
        (match_representation, match_dim) = self.bilateral_match_func(
                                                in_question_repres, 
                                                in_passage_repres,
                                                self.question_lengths, 
                                                self.passage_lengths, 
                                                question_mask, 
                                                mask, 
                                                input_dim)

        #========Prediction Layer=========
        # match_dim = 4 * self.options.aggregation_lstm_dim
        w_0 = tf.get_variable("w_0", [match_dim, match_dim / 2], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim / 2], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim / 2, self.config.num_classes],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [self.config.num_classes],dtype=tf.float32)

        # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.tanh(logits)
        logits = tf.nn.dropout(logits, (1 - self.dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1

        # 预测概率分布与损失
        self.y_hat, self.total_loss = self.add_loss_op(logits)
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
        self.dropout_rate = 1 - self.keep_prob

        self.question_lengths = tf.placeholder(tf.int32, [None])
        self.passage_lengths = tf.placeholder(tf.int32, [None])

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
        out1 = self.mlp(seq, self.config.context_lstm_dim, 1, 
                    tf.nn.sigmoid, name + '_sigmoid', reuse=reuse)
        out2 = self.mlp(seq, self.config.context_lstm_dim, 1, 
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
        # tf.add_to_collection('total_loss', loss)
        # total_loss = tf.add_n(tf.get_collection('total_loss'))
        return y_hat, loss

    def add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            train_variables = tf.trainable_variables()
            # 加入l2 loss
            if self.config.lambda_l2 > 0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_variables if v.get_shape().ndims > 1])
                loss = loss + self.options.lambda_l2 * l2_loss

            # 记录训练步骤
            self.global_step = tf.Variable(0, 
                    name='global_step', trainable=False)

            opt = tf.train.AdamOptimizer(self.config.lr)
            # train_op = opt.minimize(loss, self.global_step)
            # 梯度剪枝
            grads_vars = opt.compute_gradients(loss, train_variables)
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (
                    tf.clip_by_norm(grad, self.config.grad_clip), var)
            train_op = opt.apply_gradients(
                grads_vars, global_step=self.global_step)
            # 变量滑动平均
            if self.config.with_moving_average:
                # Track the moving averages of all trainable variables.
                MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
                variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                train_ops = [train_op, variables_averages_op]
                train_op = tf.group(*train_ops)
            return train_op

    def bilateral_match_func(self, in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, 
                        passage_mask, input_dim):
        question_aware_representatins = []
        question_aware_dim = 0
        passage_aware_representatins = []
        passage_aware_dim = 0

        # ====word level matching======
        (match_reps, match_dim) = self.match_passage_with_question(in_passage_repres, 
                                    in_question_repres, passage_mask, question_mask, 
                                    passage_lengths,
                                    question_lengths, input_dim, scope="word_match_forward",
                                    with_full_match=False, with_maxpool_match=self.config.with_maxpool_match,
                                    with_attentive_match=self.config.with_attentive_match,
                                    with_max_attentive_match=self.config.with_max_attentive_match, 
                                    dropout_rate=self.dropout_rate, forward=True)
        question_aware_representatins.append(match_reps)
        question_aware_dim += match_dim

        (match_reps, match_dim) = self.match_passage_with_question(in_question_repres, 
                                    in_passage_repres, question_mask, passage_mask, 
                                    question_lengths,
                                    passage_lengths, input_dim, scope="word_match_backward",
                                    with_full_match=False, with_maxpool_match=self.config.with_maxpool_match,
                                    with_attentive_match=self.config.with_attentive_match,
                                    with_max_attentive_match=self.config.with_max_attentive_match,
                                    dropout_rate=self.dropout_rate, forward=False)
        passage_aware_representatins.append(match_reps)
        passage_aware_dim += match_dim

        with tf.variable_scope('context_MP_matching'):
            for i in xrange(self.config.context_layer_num): # support multiple context layer
                with tf.variable_scope('layer-{}'.format(i)):
                    # contextual lstm for both passage and question
                    in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
                    in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(passage_mask, axis=-1))
                    (question_context_representation_fw, question_context_representation_bw,
                     in_question_repres) = layer_utils.my_lstm_layer(
                            in_question_repres, self.config.context_lstm_dim, input_lengths=question_lengths, scope_name="context_represent",
                            reuse=False, dropout_rate=self.dropout_rate, use_cudnn=self.config.use_cudnn)
                    (passage_context_representation_fw, passage_context_representation_bw, 
                     in_passage_repres) = layer_utils.my_lstm_layer(
                            in_passage_repres, self.config.context_lstm_dim, input_lengths=passage_lengths, scope_name="context_represent",
                            reuse=True, dropout_rate=self.dropout_rate, use_cudnn=self.config.use_cudnn)

                    # Multi-perspective matching
                    with tf.variable_scope('left_MP_matching'):
                        (match_reps, match_dim) = self.match_passage_with_question(passage_context_representation_fw,
                                    question_context_representation_fw, passage_mask, question_mask, passage_lengths,
                                    question_lengths, self.config.context_lstm_dim, scope="forward_match",
                                    with_full_match=self.config.with_full_match, with_maxpool_match=self.config.with_maxpool_match,
                                    with_attentive_match=self.config.with_attentive_match,
                                    with_max_attentive_match=self.config.with_max_attentive_match,
                                    dropout_rate=self.dropout_rate, forward=True)
                        question_aware_representatins.append(match_reps)
                        question_aware_dim += match_dim
                        (match_reps, match_dim) = self.match_passage_with_question(passage_context_representation_bw,
                                    question_context_representation_bw, passage_mask, question_mask, passage_lengths,
                                    question_lengths, self.config.context_lstm_dim, scope="backward_match",
                                    with_full_match=self.config.with_full_match, with_maxpool_match=self.config.with_maxpool_match,
                                    with_attentive_match=self.config.with_attentive_match,
                                    with_max_attentive_match=self.config.with_max_attentive_match,
                                    dropout_rate=self.dropout_rate, forward=False)
                        question_aware_representatins.append(match_reps)
                        question_aware_dim += match_dim

                    with tf.variable_scope('right_MP_matching'):
                        (match_reps, match_dim) = self.match_passage_with_question(question_context_representation_fw,
                                    passage_context_representation_fw, question_mask, passage_mask, question_lengths,
                                    passage_lengths, self.config.context_lstm_dim, scope="forward_match",
                                    with_full_match=self.config.with_full_match, with_maxpool_match=self.config.with_maxpool_match,
                                    with_attentive_match=self.config.with_attentive_match,
                                    with_max_attentive_match=self.config.with_max_attentive_match,
                                    dropout_rate=self.dropout_rate, forward=True)
                        passage_aware_representatins.append(match_reps)
                        passage_aware_dim += match_dim
                        (match_reps, match_dim) = self.match_passage_with_question(question_context_representation_bw,
                                    passage_context_representation_bw, question_mask, passage_mask, question_lengths,
                                    passage_lengths, self.config.context_lstm_dim, scope="backward_match",
                                    with_full_match=self.config.with_full_match, with_maxpool_match=self.config.with_maxpool_match,
                                    with_attentive_match=self.config.with_attentive_match,
                                    with_max_attentive_match=self.config.with_max_attentive_match,
                                    dropout_rate=self.dropout_rate, forward=False)
                        passage_aware_representatins.append(match_reps)
                        passage_aware_dim += match_dim

        question_aware_representatins = tf.concat(axis=2, values=question_aware_representatins) # [batch_size, passage_len, passage_aware_dim]
        passage_aware_representatins = tf.concat(axis=2, values=passage_aware_representatins) # [batch_size, question_len, question_aware_dim]

        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - self.dropout_rate))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - self.dropout_rate))
            
        # ======Highway layer======
        if self.config.with_match_highway:
            with tf.variable_scope("left_matching_highway"):
                question_aware_representatins = self.multi_highway_layer(question_aware_representatins, question_aware_dim,
                                                                    self.config.highway_layer_num)
            with tf.variable_scope("right_matching_highway"):
                passage_aware_representatins = self.multi_highway_layer(passage_aware_representatins, passage_aware_dim,
                                                               self.config.highway_layer_num)

        #========Aggregation Layer======
        aggregation_representation = []
        aggregation_dim = 0
        
        qa_aggregation_input = question_aware_representatins
        pa_aggregation_input = passage_aware_representatins
        with tf.variable_scope('aggregation_layer'):
            for i in xrange(self.config.aggregation_layer_num): # support multiple aggregation layer
                qa_aggregation_input = tf.multiply(qa_aggregation_input, tf.expand_dims(passage_mask, axis=-1))
                (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                            qa_aggregation_input, self.config.aggregation_lstm_dim, 
                            input_lengths=passage_lengths, scope_name='left_layer-{}'.format(i),
                            reuse=False, dropout_rate=self.dropout_rate, use_cudnn=self.config.use_cudnn)
                fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, passage_lengths - 1)
                bw_rep = bw_rep[:, 0, :]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2 * self.config.aggregation_lstm_dim
                # [batch_size, question_len, 2*aggregation_lstm_dim]
                qa_aggregation_input = cur_aggregation_representation

                pa_aggregation_input = tf.multiply(pa_aggregation_input, tf.expand_dims(question_mask, axis=-1))
                (fw_rep, bw_rep, cur_aggregation_representation) = layer_utils.my_lstm_layer(
                            pa_aggregation_input, self.config.aggregation_lstm_dim,
                            input_lengths=question_lengths, scope_name='right_layer-{}'.format(i),
                            reuse=False, dropout_rate=self.dropout_rate, use_cudnn=self.config.use_cudnn)
                fw_rep = layer_utils.collect_final_step_of_lstm(fw_rep, question_lengths - 1)
                bw_rep = bw_rep[:, 0, :]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2 * self.config.aggregation_lstm_dim
                # [batch_size, passage_len, 2*aggregation_lstm_dim]
                pa_aggregation_input = cur_aggregation_representation

        # [batch_size, 4*aggregation_lstm_dim*aggregation_layer_num]
        aggregation_representation = tf.concat(axis=1, values=aggregation_representation)

        # ======Highway layer======
        if self.config.with_aggregation_highway:
            with tf.variable_scope("aggregation_highway"):
                agg_shape = tf.shape(aggregation_representation)
                batch_size = agg_shape[0]
                aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
                aggregation_representation = self.multi_highway_layer(aggregation_representation, aggregation_dim, self.config.highway_layer_num)
                aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
        return (aggregation_representation, aggregation_dim)

    def match_passage_with_question(self, passage_reps, question_reps, 
                                    passage_mask, question_mask, 
                                    passage_lengths, question_lengths,
                                    context_lstm_dim, scope=None,
                                    with_full_match=True, with_maxpool_match=True, 
                                    with_attentive_match=True, with_max_attentive_match=True,
                                    dropout_rate=0, forward=True):
        passage_reps = tf.multiply(passage_reps, tf.expand_dims(passage_mask,-1))
        question_reps = tf.multiply(question_reps, tf.expand_dims(question_mask,-1))
        all_question_aware_representatins = []
        dim = 0
        with tf.variable_scope(scope or "match_passage_with_question"):
            # relevancy_matrix: [batch_size, p_len, q_len]
            relevancy_matrix = self.cal_relevancy_matrix(question_reps, passage_reps)
            relevancy_matrix = self.mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask)

            all_question_aware_representatins.append(tf.reduce_max(relevancy_matrix, axis=2, keep_dims=True))
            all_question_aware_representatins.append(tf.reduce_mean(relevancy_matrix, axis=2, keep_dims=True))
            dim += 2
            if with_full_match:
                if forward:
                    question_full_rep = layer_utils.collect_final_step_of_lstm(question_reps, question_lengths - 1)
                else:
                    question_full_rep = question_reps[:,0,:]

                passage_len = tf.shape(passage_reps)[1]
                question_full_rep = tf.expand_dims(question_full_rep, axis=1)
                # [batch_size, pasasge_len, feature_dim]
                question_full_rep = tf.tile(question_full_rep, [1, passage_len, 1])
                # attentive_rep: [batch_size, passage_len, match_dim]
                (attentive_rep, match_dim) = self.multi_perspective_match(context_lstm_dim,
                                    passage_reps, question_full_rep, 
                                    dropout_rate=self.dropout_rate,
                                    scope_name='mp-match-full-match')
                all_question_aware_representatins.append(attentive_rep)
                dim += match_dim

            if with_maxpool_match:
                maxpooling_decomp_params = tf.get_variable("maxpooling_matching_decomp",
                                                shape=[self.config.cosine_MP_dim, context_lstm_dim], 
                                                dtype=tf.float32)
                # maxpooling_rep: [batch_size, passage_len, 2 * cosine_MP_dim]
                maxpooling_rep = self.cal_maxpooling_matching(passage_reps, 
                                    question_reps, maxpooling_decomp_params)
                all_question_aware_representatins.append(maxpooling_rep)
                dim += 2 * self.config.cosine_MP_dim

            if with_attentive_match:
                # atten_scores: [batch_size, p_len, q_len]
                atten_scores = layer_utils.calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
                        scope_name="attention", att_type=self.config.att_type, att_dim=self.config.att_dim,
                        remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, dropout_rate=self.dropout_rate)
                att_question_contexts = tf.matmul(atten_scores, question_reps)
                (attentive_rep, match_dim) = self.multi_perspective_match(context_lstm_dim,
                        passage_reps, att_question_contexts, dropout_rate=self.dropout_rate,
                        scope_name='mp-match-att_question')
                all_question_aware_representatins.append(attentive_rep)
                dim += match_dim

            if with_max_attentive_match:
                # relevancy_matrix: [batch_size, p_len, q_len]
                # question_reps: [batch_size, q_len, dim]
                # max_att: [batch_size, p_len, dim]
                max_att = self.cal_max_question_representation(question_reps, relevancy_matrix)
                # max_attentive_rep: [batch_size, passage_len, match_dim]
                (max_attentive_rep, match_dim) = self.multi_perspective_match(context_lstm_dim,
                        passage_reps, max_att, dropout_rate=self.dropout_rate,
                        scope_name='mp-match-max-att')
                all_question_aware_representatins.append(max_attentive_rep)
                dim += match_dim

            all_question_aware_representatins = tf.concat(axis=2, values=all_question_aware_representatins)
        return (all_question_aware_representatins, dim)


    def multi_perspective_match(self, feature_dim, 
                                repres1, repres2, 
                                dropout_rate=0.2,
                                scope_name='mp-match', 
                                reuse=False):
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
            if self.config.with_cosine:
                cosine_value = layer_utils.cosine_distance(repres1, repres2, cosine_norm=False)
                cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
                matching_result.append(cosine_value)
                match_dim += 1

            if self.config.with_mp_cosine:
                mp_cosine_params = tf.get_variable("mp_cosine", shape=[self.config.cosine_MP_dim, feature_dim], dtype=tf.float32)
                mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
                mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
                repres1_flat = tf.expand_dims(repres1, axis=2)
                repres2_flat = tf.expand_dims(repres2, axis=2)
                # mp_cosine_matching: [batch_size, len, mp_dim]
                mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(repres1_flat, mp_cosine_params),
                                                                 repres2_flat, cosine_norm=False)
                matching_result.append(mp_cosine_matching)
                match_dim += self.config.cosine_MP_dim

        matching_result = tf.concat(axis=2, values=matching_result)
        return (matching_result, match_dim)

    def cal_relevancy_matrix(self, in_question_repres, in_passage_repres):
        in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
        in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
        relevancy_matrix = layer_utils.cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
        return relevancy_matrix

    def mask_relevancy_matrix(self, relevancy_matrix, question_mask, passage_mask):
        # relevancy_matrix: [batch_size, passage_len, question_len]
        # question_mask: [batch_size, question_len]
        # passage_mask: [batch_size, passsage_len]
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
        return relevancy_matrix

    def cal_max_question_representation(self, 
                                        question_representation, 
                                        atten_scores):
        # question_representation: [batch_size, q_len, dim]
        # atten_scores: [batch_size, passage_len, q_len]
        atten_positions = tf.argmax(atten_scores, axis=2, 
                            output_type=tf.int32)
        max_question_reps = layer_utils.collect_representation(
                                question_representation, atten_positions)
        # [batch_size, passage_len, dim]
        return max_question_reps

    def cal_maxpooling_matching(self, passage_rep, 
                                question_rep, decompose_params):
        # passage_representation: [batch_size, passage_len, dim]
        # qusetion_representation: [batch_size, question_len, dim]
        # decompose_params: [decompose_dim, dim]
        def multi_perspective_expand_for_2D(in_tensor, decompose_params):
            in_tensor = tf.expand_dims(in_tensor, axis=1) #[batch_size, 'x', dim]
            decompose_params = tf.expand_dims(decompose_params, axis=0) # [1, decompse_dim, dim]
            return tf.multiply(in_tensor, decompose_params) # [batch_size, decompse_dim, dim]

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

    def highway_layer(self, in_val, output_size, scope=None):
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

    def multi_highway_layer(self, in_val, output_size, num_layers, scope=None):
        scope_name = 'highway_layer'
        if scope is not None: scope_name = scope
        for i in xrange(num_layers):
            cur_scope_name = scope_name + "-{}".format(i)
            in_val = self.highway_layer(in_val, output_size, scope=cur_scope_name)
        return in_val


