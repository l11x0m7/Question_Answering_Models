
# coding: utf-8

# In[1]:


import os
import sys
stdout = sys.stdout
reload(sys)
sys.stdout = stdout

import cPickle as pkl

from collections import Counter
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import jieba
# jieba.enable_parallel(8)
lemma = WordNetLemmatizer()

raw_data_path = './data/WikiQA/raw'
processed_data_path = './data/WikiQA/processed'

if not os.path.exists(processed_data_path):
    os.mkdir(processed_data_path)


# In[8]:


# 分词、词干化处理
def segment(filename, use_lemma=True):
    processed_qa = []
    count = 0
    with open(os.path.join(raw_data_path, filename), 'r') as fr:
        fr.readline()
        for line in fr:
            items = line.strip().split('\t')
            qid, q, aid, a, label = items[0], items[1], items[4], items[5], items[6]
            if use_lemma:
                q = ' '.join([lemma.lemmatize(_) for _ in jieba.cut(q)]).lower()
                a = ' '.join([lemma.lemmatize(_) for _ in jieba.cut(a)]).lower()
            else:
                q = ' '.join(jieba.cut(q)).lower()
                q = ' '.join(jieba.cut(a)).lower()
            processed_qa.append('\t'.join([qid, q, aid, a, label]))
            count += 1
            if count % 1000 == 0:
                print('Finished {}'.format(count))
    return processed_qa

# 构建词典
def build_vocab(corpus, topk=None):
    vocab = Counter()
    for line in corpus:
        qid, q, aid, a, label = line.strip().split('\t')
        vocab.update(q.split())
        vocab.update(a.split())
    if topk:
        vocab = vocab.most_common(topk)
    else:
        vocab = dict(vocab.most_common()).keys()
    vocab = {_ : i+2 for i, _ in enumerate(vocab)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    reverse_vocab = dict(zip(vocab.values(), vocab.keys()))
    return vocab, reverse_vocab

# 将每个词映射为词典中的id
def transform(corpus, word2id, unk_id=1):
    transformed_corpus = []
    for line in corpus:
        qid, q, aid, a, label = line.strip().split('\t')
        q = [word2id.get(w, unk_id) for w in q.split()]
        a = [word2id.get(w, unk_id) for w in a.split()]
        transformed_corpus.append([qid, q, aid, a, int(label)])
    return transformed_corpus

# 得到pointwise形式的数据，即(Q, A, label)
def pointwise_data(corpus, keep_ids=False):
    # (q, a, label)
    pointwise_corpus = []
    for sample in corpus:
        qid, q, aid, a, label = sample
        if keep_ids:
            pointwise_corpus.append((qid, q, aid, a, label))
        else:
            pointwise_corpus.append((q, a, label))
    return pointwise_corpus

# 得到pairwise形式的数据，即(Q, positive A, negative A)
def pairwise_data(corpus):
    # (q, a_pos, a_neg), two answers must from the same q
    # once a question contains no positive answers, we discard this sample.
    pairwise_corpus = dict()
    for sample in corpus:
        qid, q, aid, a, label = sample
        pairwise_corpus.setdefault(qid, dict())
        pairwise_corpus[qid].setdefault('pos', list())
        pairwise_corpus[qid].setdefault('neg', list())
        pairwise_corpus[qid]['q'] = q
        if label == 0:
            pairwise_corpus[qid]['neg'].append(a)
        else:
            pairwise_corpus[qid]['pos'].append(a)
    real_pairwise_corpus = []
    for qid in pairwise_corpus:
        q = pairwise_corpus[qid]['q']
        for pos in pairwise_corpus[qid]['pos']:
            for neg in pairwise_corpus[qid]['neg']:
                real_pairwise_corpus.append((q, pos, neg))
    return real_pairwise_corpus
    
# 得到listwise形式的数据，即(Q, All answers related to this Q)
def listwise_data(corpus):
    # (q, a_list)
    listwise_corpus = dict()
    for sample in corpus:
        qid, q, aid, a, label = sample
        listwise_corpus.setdefault(qid, dict())
        listwise_corpus[qid].setdefault('a', list())
        listwise_corpus[qid]['q'] = q            
        listwise_corpus[qid]['a'].append(a)
    real_listwise_corpus = []
    for qid in listwise_corpus:
        q = listwise_corpus[qid]['q']
        alist = listwise_corpus[qid]['a']
        real_listwise_corpus.append((q, alist))
    return real_listwise_corpus


train_processed_qa = segment('WikiQA-train.tsv')
val_processed_qa = segment('WikiQA-dev.tsv')
test_processed_qa = segment('WikiQA-test.tsv')
word2id, id2word = build_vocab(train_processed_qa)

transformed_train_corpus = transform(train_processed_qa, word2id)
pointwise_train_corpus = pointwise_data(transformed_train_corpus, keep_ids=True)
pairwise_train_corpus = pairwise_data(transformed_train_corpus)
listwise_train_corpus = listwise_data(transformed_train_corpus)

transformed_val_corpus = transform(val_processed_qa, word2id)
pointwise_val_corpus = pointwise_data(transformed_val_corpus, keep_ids=True)
pairwise_val_corpus = pointwise_data(transformed_val_corpus, keep_ids=True)
listwise_val_corpus = listwise_data(transformed_val_corpus)

transformed_test_corpus = transform(test_processed_qa, word2id)
pointwise_test_corpus = pointwise_data(transformed_test_corpus, keep_ids=True)
pairwise_test_corpus = pointwise_data(transformed_test_corpus, keep_ids=True)
listwise_test_corpus = listwise_data(transformed_test_corpus)


with open(os.path.join(processed_data_path, 'vocab.pkl'), 'w') as fw:
    pkl.dump([word2id, id2word], fw)
with open(os.path.join(processed_data_path, 'pointwise_corpus.pkl'), 'w') as fw:
    pkl.dump([pointwise_train_corpus, pointwise_val_corpus, pointwise_test_corpus], fw)
with open(os.path.join(processed_data_path, 'pairwise_corpus.pkl'), 'w') as fw:
    pkl.dump([pairwise_train_corpus, pairwise_val_corpus, pairwise_test_corpus], fw)
with open(os.path.join(processed_data_path, 'listwise_corpus.pkl'), 'w') as fw:
    pkl.dump([listwise_train_corpus, listwise_val_corpus, listwise_test_corpus], fw)
    
print('done!')

