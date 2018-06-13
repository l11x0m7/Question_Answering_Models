# -*- encoding:utf8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
from copy import deepcopy
stdout = sys.stdout
reload(sys)
sys.stdout = stdout

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cPickle as pkl
from utils import *
from models import SeqMatchSeq


class SeqMatchSeqConfig(object):
    def __init__(self, vocab_size, embeddings=None):
        # 输入问题(句子)长度
        self.max_q_length = 200
        # 输入答案长度
        self.max_a_length = 200
        # 循环数
        self.num_epochs = 100
        # batch大小
        self.batch_size = 128
        # 词表大小
        self.vocab_size = vocab_size
        # 词向量大小
        self.embeddings = embeddings
        self.embedding_size = 100
        if self.embeddings is not None:
            self.embedding_size = embeddings.shape[1]
        # keep_prob=1-dropout
        self.keep_prob = 0.6
        # 学习率
        self.lr = 0.0003
        self.grad_clip = 1

        self.reg = 0
        self.mem_dim = 128
        self.cov_dim = 128
        self.filter_sizes = [2, 3, 4, 5]
        self.comp_type = 'mul'

        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2


def train(train_corpus, config, val_corpus, eval_train_corpus=None):
    iterator = Iterator(train_corpus)

    with tf.Session(config=config.cf) as sess:
        model = SeqMatchSeq(config)
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        for epoch in xrange(config.num_epochs):
            count = 0
            for batch_x in iterator.next(config.batch_size, shuffle=True):
                batch_qids, batch_q, batch_aids, batch_ap, labels = zip(*batch_x)
                batch_q = np.asarray(batch_q)
                batch_ap = np.asarray(batch_ap)
                labels = np.asarray(labels).astype(np.int32)
                _, loss = sess.run([model.train_op, model.total_loss], 
                                   feed_dict={model.q:batch_q, 
                                              model.a:batch_ap,
                                              model.y:labels,
                                              model.keep_prob:config.keep_prob})
                count += 1
                if count % 10 == 0:
                    print('[epoch {}, batch {}]Loss:{}'.format(epoch, count, loss))
            saver.save(sess,'{}/my_model'.format(model_path), global_step=epoch)
            if eval_train_corpus is not None:
                train_res = evaluate(sess, model, eval_train_corpus, config)
                print('[train] ' + train_res)
            if val_corpus is not None:
                val_res = evaluate(sess, model, val_corpus, config)
                print('[eval] ' + val_res)


def evaluate(sess, model, corpus, config):
    iterator = Iterator(corpus)

    count = 0
    total_qids = []
    total_aids = []
    total_pred = []
    total_labels = []
    total_loss = 0.
    for batch_x in iterator.next(config.batch_size, shuffle=False):
        batch_qids, batch_q, batch_aids, batch_ap, labels = zip(*batch_x)
        batch_q = np.asarray(batch_q)
        batch_ap = np.asarray(batch_ap)
        y_hat, loss = sess.run([model.y_hat, model.total_loss], 
                           feed_dict={model.q:batch_q, 
                                      model.a:batch_ap, 
                                      model.y:labels,
                                      model.keep_prob:1.})
        y_hat = np.argmax(y_hat, axis=-1)
        total_loss += loss
        count += 1
        total_qids.append(batch_qids)
        total_aids.append(batch_aids)
        total_pred.append(y_hat)
        total_labels.append(labels)
        # print(batch_qids[0], [id2word[_] for _ in batch_q[0]], 
        #     batch_aids[0], [id2word[_] for _ in batch_ap[0]])
    total_qids = np.concatenate(total_qids, axis=0)
    total_aids = np.concatenate(total_aids, axis=0)
    total_pred = np.concatenate(total_pred, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    MAP, MRR = eval_map_mrr(total_qids, total_aids, total_pred, total_labels)
    # print('Eval loss:{}'.format(total_loss / count))
    return 'MAP:{}, MRR:{}'.format(MAP, MRR)
                

def test(corpus, config):
    with tf.Session(config=config.cf) as sess:
        model = SeqMatchSeq(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print('[test] ' + evaluate(sess, model, corpus, config))
                    

def main(args):
    max_q_length = 30
    max_a_length = 100

    with open(os.path.join(processed_data_path, 'pointwise_corpus.pkl'), 'r') as fr:
        train_corpus, val_corpus, test_corpus = pkl.load(fr)

    embeddings = build_embedding(embedding_path, word2id)
    
    train_qids, train_q, train_aids, train_ap, train_labels = zip(*train_corpus)
    train_q = padding(train_q, max_q_length)
    train_ap = padding(train_ap, max_a_length)
    train_corpus = zip(train_qids, train_q, train_aids, train_ap, train_labels)


    val_qids, val_q, val_aids, val_ap, labels = zip(*val_corpus)
    val_q = padding(val_q, max_q_length)
    val_ap = padding(val_ap, max_a_length)
    val_corpus = zip(val_qids, val_q, val_aids, val_ap, labels)


    test_qids, test_q, test_aids, test_ap, labels = zip(*test_corpus)
    test_q = padding(test_q, max_q_length)
    test_ap = padding(test_ap, max_a_length)
    test_corpus = zip(test_qids, test_q, test_aids, test_ap, labels)

    config = SeqMatchSeqConfig(max(word2id.values()) + 1, embeddings=embeddings)
    config.max_q_length = max_q_length
    config.max_a_length = max_a_length
    if args.train:
        train(deepcopy(train_corpus), config, val_corpus, deepcopy(train_corpus))
    elif args.test:
        test(test_corpus, config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  help="whether to train", action='store_true')
    parser.add_argument("--test",  help="whether to test", action='store_true')
    args = parser.parse_args()

    raw_data_path = '../data/WikiQA/raw'
    processed_data_path = '../data/WikiQA/processed'
    embedding_path = '../data/embedding/glove.6B.300d.txt'
    model_path = 'models'

    with open(os.path.join(processed_data_path, 'vocab.pkl'), 'r') as fr:
        word2id, id2word = pkl.load(fr)
    main(args)
