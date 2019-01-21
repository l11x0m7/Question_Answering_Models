import tensorflow as tf
from layers import dropout, native_gru, cudnn_gru, ptr_layer, summ, softmax_mask, dense
from layers import dot_attention
from loss import GHMC_loss, GHMR_loss


class Hybrid(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.trainable = trainable
        self.global_step = tf.get_variable(
            "global_step", shape=[], dtype=tf.int32, 
            initializer=tf.zeros_initializer(), trainable=False)
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable(
            "word_mat", dtype=tf.float32, initializer=tf.constant(word_mat, dtype=tf.float32), 
            trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", dtype=tf.float32, initializer=tf.constant(char_mat, dtype=tf.float32),
            trainable=True)
        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            N, CL = config.batch_size, config.char_limit
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
            self.c_maxlen = config.para_limit
            self.q_maxlen = config.ques_limit

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.forward()

        if self.trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def contextual_encoding(self, name, doc, seq_length, hidden_size, keep_prob, bidir=True):
        with tf.variable_scope(name):
            if bidir:
                fw_final = tf.contrib.rnn.GRUCell(hidden_size)
                bk_final = tf.contrib.rnn.GRUCell(hidden_size)
                (fw_states, bk_states), _ = tf.nn.bidirectional_dynamic_rnn(
                    fw_final, bk_final, doc, sequence_length=tf.to_int32(seq_length),
                    dtype=tf.float32)
                contextual_emb = tf.concat([fw_states, bk_states], axis=2)
                contextual_emb = dropout(contextual_emb, keep_prob=keep_prob, is_train=self.is_train)
            else:
                fw_final = tf.contrib.rnn.GRUCell(hidden_size)
                fw_states, _ = tf.nn.dynamic_rnn(
                    fw_final, doc, sequence_length=tf.to_int32(seq_length),
                    dtype=tf.float32)
                contextual_emb = fw_states
                contextual_emb = dropout(contextual_emb, keep_prob=keep_prob, is_train=self.is_train)
            return contextual_emb

    def forward(self):
        # in: c, q, c_mask, q_mask, ch, qh, y1, y2
        # out: yp1, yp2, loss
        config = self.config
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope('emb'):
            with tf.variable_scope('char'):
                ch_emb = tf.reshape(
                    tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc])
                qh_emb = tf.reshape(
                    tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])
                ch_emb = dropout(ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                qh_emb = dropout(qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                cell_fw = tf.contrib.rnn.GRUCell(dg)
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
                ch_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                qh_emb = tf.concat([state_fw, state_bw], axis=1)
                qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])
            with tf.variable_scope('word'):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            c_emb_ori = tf.concat([c_emb, ch_emb], axis=2)
            q_emb_ori = tf.concat([q_emb, qh_emb], axis=2)

            # spatial dropout
            if config.use_spatial_dp:
                print("Using spatial dropout\n")
                if self.is_train:
                    q_emb_shape = tf.shape(q_emb_ori)
                    c_emb_shape = tf.shape(c_emb_ori)
                    q_emb = tf.nn.dropout(q_emb_ori, keep_prob=0.5 + config.keep_prob / 2, noise_shape=(q_emb_shape[0], 1, q_emb_shape[2]))
                    c_emb = tf.nn.dropout(c_emb_ori, keep_prob=0.5 + config.keep_prob / 2, noise_shape=(c_emb_shape[0], 1, c_emb_shape[2]))
                else:
                    q_emb = q_emb_ori
                    c_emb = c_emb_ori
            else:
                c_emb = c_emb_ori
                q_emb = q_emb_ori

        # context encoding: method1
        with tf.variable_scope('encoding'):
            rnn = gru(num_layers=3, num_units=d, 
                batch_size=N, input_size=c_emb.get_shape().as_list()[-1], 
                keep_prob=config.keep_prob, is_train=self.is_train)
            c = rnn(c_emb, seq_len=self.c_len, concat=True, keep_origin_input=True)
            q = rnn(q_emb, seq_len=self.q_len, concat=True, keep_origin_input=True)

        with tf.variable_scope('attention'):
            qc_att = dot_attention(inputs=c, memory=q, 
                                   hidden_size=d, mask=self.q_mask, 
                                   keep_prob=config.keep_prob, 
                                   is_train=self.is_train, scope='qc_dot_att')
            rnn = gru(num_layers=1, num_units=d, 
                batch_size=N, input_size=qc_att.get_shape().as_list()[-1], 
                keep_prob=config.keep_prob, is_train=self.is_train, scope='qc')
            
            cq_att = dot_attention(inputs=q, memory=c, 
                                   hidden_size=d, mask=self.c_mask, 
                                   keep_prob=config.keep_prob, 
                                   is_train=self.is_train, scope='cq_dot_att')
            rnn = gru(num_layers=1, num_units=d, 
                batch_size=N, input_size=cq_att.get_shape().as_list()[-1], 
                keep_prob=config.keep_prob, is_train=self.is_train, scope='cq')
            c = rnn(qc_att, seq_len=self.c_len, keep_origin_input=False)
            q = rnn(cq_att, seq_len=self.q_len, keep_origin_input=False)

        # seq_length = self.q_len
        # idx = tf.concat(
        #         [tf.expand_dims(tf.range(tf.shape(q)[0]), axis=1),
        #          tf.expand_dims(seq_length - 1, axis=1)], axis=1)
        # # (B, 2h)
        # q_state = tf.gather_nd(q, idx)

        with tf.variable_scope('hybrid'):
            # B * N * Q
            doc_qry_mask = tf.keras.backend.batch_dot(tf.expand_dims(tf.cast(self.c_mask, tf.float32), 2), tf.expand_dims(tf.cast(self.q_mask, tf.float32), 1), axes=[2, 1])
            # (B, D, Q, 2h)
            doc_expand_embed = tf.tile(tf.expand_dims(c, 2), [1, 1, self.q_maxlen, 1])
            # (B, D, Q, 2h)
            qry_expand_embed = tf.tile(tf.expand_dims(q, 1), [1, self.c_maxlen, 1, 1])
            doc_qry_dot_embed = doc_expand_embed * qry_expand_embed
            # (B, D, Q, 6h)
            doc_qry_embed = tf.concat([doc_expand_embed, qry_expand_embed, doc_qry_dot_embed], axis=3)
            # attention way
            num_units = doc_qry_embed.shape[-1]
            with tf.variable_scope('bi_attention'):
                w = tf.get_variable('W_att', shape=(num_units, 1), 
                        dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
                # (B, D, Q)
                S = tf.matmul(tf.reshape(doc_qry_embed, (-1, doc_qry_embed.shape[-1])), w)
                S = tf.reshape(S, (N, self.c_maxlen, self.q_maxlen))
                # context2query, (B, D, 2h)
                c2q = tf.keras.backend.batch_dot(tf.nn.softmax(softmax_mask(S, doc_qry_mask), dim=2), q)
                c2q_gated = c2q * c
                

            with tf.variable_scope('gated_attention'):
                # Gated Attention
                g_doc_qry_att = tf.keras.backend.batch_dot(c, tf.transpose(q, (0, 2, 1)))
                # B * N * Q
                alphas = tf.nn.softmax(softmax_mask(g_doc_qry_att, doc_qry_mask), dim=2)

                q_rep = tf.keras.backend.batch_dot(alphas, q) # B x N x 2D
                d_gated = c * q_rep

                G = tf.concat([c, c2q, q_rep, c2q_gated, d_gated], axis=-1)
                # G = tf.nn.relu(dense(G, d * 2))

            with tf.variable_scope('match'):
                G = dot_attention(inputs=G, memory=G, 
                                       hidden_size=d, mask=self.c_mask, 
                                       keep_prob=config.keep_prob, 
                                       is_train=self.is_train)

                rnn = gru(num_layers=1, num_units=d, 
                    batch_size=N, input_size=G.get_shape().as_list()[-1], 
                    keep_prob=config.keep_prob, is_train=self.is_train)
                doc_encoding = rnn(G, seq_len=self.c_len, concat=False)

        with tf.variable_scope('pointer'):
            # Use self-attention or bilinear attention
            init = summ(q, d, mask=self.q_mask, 
                        keep_prob=config.keep_prob, is_train=self.is_train)

            # init = self.bilinear_attention_layer(c, q_state, self.c_mask)
            
            pointer = ptr_layer(batch_size=N, 
                                hidden_size=init.get_shape().as_list()[-1], 
                                keep_prob=config.keep_prob, 
                                is_train=self.is_train)
            logits1, logits2 = pointer(init, doc_encoding, d, self.c_mask)

        with tf.variable_scope('predict'):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2), 
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

            # loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(
            #         logits=logits1, labels=tf.stop_gradient(self.y1))
            # loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(
            #         logits=logits2, labels=tf.stop_gradient(self.y2))
            if config.use_ghmc_or_ghmr == 'ghmc':
                print('Using GHMC Loss\n')
                ghmc_loss_func = GHMC_loss(momentum=0.)
                loss1 = ghmc_loss_func(logits1, tf.stop_gradient(self.y1))
                loss2 = ghmc_loss_func(logits2, tf.stop_gradient(self.y2))
            elif config.use_ghmc_or_ghmr == 'ghmr':
                print('Using GHMR Loss\n')
                ghmr_loss_func = GHMR_loss()
                loss1 = ghmr_loss_func(logits1, tf.stop_gradient(self.y1))
                loss2 = ghmr_loss_func(logits2, tf.stop_gradient(self.y2))
            else:
                loss1 = tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits1, labels=tf.stop_gradient(self.y1))
                loss2 = tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits2, labels=tf.stop_gradient(self.y2))
            self.loss = tf.reduce_mean(loss1 + loss2)

    def bilinear_attention_layer(self, document, query, doc_mask):
        """ can be used here to replace summ operation
        # document: (B, D, 2h)
        # query: (B, 2h)

        # return: (B, 2h)
        """
        num_units = int(document.shape[-1])
        with tf.variable_scope('bilinear_att') as vs:
            W_att = tf.get_variable('W_bilinear', shape=(num_units, num_units), 
                    dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.01, 0.01))
            M = tf.expand_dims(tf.matmul(query, W_att), axis=1)
            alpha = tf.nn.softmax(softmax_mask(tf.reduce_sum(document * M, axis=2), tf.to_float(doc_mask)))
            return tf.reduce_sum(document * tf.expand_dims(alpha, axis=2), axis=1)


    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

