# -*- encoding:utf-8 -*-
import numpy as np
import tensorflow as tf


def padding(data, max_len):
    return tf.keras.preprocessing.sequence.pad_sequences(data, max_len, padding='post', truncating='post')

def eval_map_mrr(qids, aids, preds, labels):
	# 衡量map指标和mrr指标
    dic = dict()
    pre_dic = dict()
    for qid, aid, pred, label in zip(qids, aids, preds, labels):
        pre_dic.setdefault(qid, [])
        pre_dic[qid].append([aid, pred, label])
    for qid in pre_dic:
        dic[qid] = sorted(pre_dic[qid], key=lambda k: k[1], reverse=True)
        aid2rank = {aid:[label, rank] for (rank, (aid, pred, label)) in enumerate(dic[qid])}
        dic[qid] = aid2rank
    # correct = 0
    # total = 0
    # for qid in dic:
    #     cur_correct = 0
    #     for aid in dic[qid]:
    #         if dic[qid][aid][0] == 1:
    #             cur_correct += 1
    #     if cur_correct > 0:
    #         correct += 1
    #     total += 1
    # print(correct * 1. / total)

    MAP = 0.0
    MRR = 0.0
    useful_q_len = 0
    for q_id in dic:
        sort_rank = sorted(dic[q_id].items(), key=lambda k: k[1][1], reverse=False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            if sort_rank[i][1][0] == 1:
                correct += 1
        if correct == 0:
            continue
        useful_q_len += 1
        correct = 0
        for i in range(len(sort_rank)):
            # compute MRR
            if sort_rank[i][1][0] == 1 and mrr_mark == False:
                MRR += 1.0 / float(i + 1)
                mrr_mark = True
            # compute MAP
            total += 1
            if sort_rank[i][1][0] == 1:
                correct += 1
                AP += float(correct) / float(total)
        
        AP /= float(correct)
        MAP += AP

    MAP /= useful_q_len
    MRR /= useful_q_len
    return MAP, MRR

def build_embedding(in_file, word_dict):
	# 构建预训练的embedding矩阵
    num_words = max(word_dict.values()) + 1
    dim = int(in_file.split('.')[-2][:-1])
    embeddings = np.zeros((num_words, dim))

    if in_file is not None:
        pre_trained = 0
        initialized = {}
        avg_sigma = 0
        avg_mu = 0
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                initialized[sp[0]] = True
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
                mu = embeddings[word_dict[sp[0]]].mean()
                #print embeddings[word_dict[sp[0]]]
                sigma = np.std(embeddings[word_dict[sp[0]]])
                avg_mu += mu
                avg_sigma += sigma
        avg_sigma /= 1. * pre_trained
        avg_mu /= 1. * pre_trained
        for w in word_dict:
            if w not in initialized:
                embeddings[word_dict[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
        print('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings.astype(np.float32)


class Iterator(object):
    """
    数据迭代器
    """
    def __init__(self, x):
        self.x = x
        self.sample_num = len(self.x)

    def next_batch(self, batch_size, shuffle=True):
        # produce X, Y_out, Y_in, X_len, Y_in_len, Y_out_len
        if shuffle:
            np.random.shuffle(self.x)
        l = np.random.randint(0, self.sample_num - batch_size + 1)
        r = l + batch_size
        x_part = self.x[l:r]
        return x_part

    def next(self, batch_size, shuffle=False):
        if shuffle:
            np.random.shuffle(self.x)
        l = 0
        while l < self.sample_num:
            r = min(l + batch_size, self.sample_num)
            batch_size = r - l
            x_part = self.x[l:r]
            l += batch_size
            yield x_part
