# Question_Answering_Models
This repo collects and re-produces models related to domains of question answering and machine reading comprehension

## comunity QA

### Dataset

WikiQA, TrecQA, InsuranceQA

#### data preprocess on WikiQA


```
bash download.sh
python preprocess_wiki.py
```

### Pointwise Style

#### Siamese-NN model

This model is a simple complementation of a Siamese NN QA model with a pointwise way.

[To this repo](./siamese_nn)

##### train model

`python siamese.py --train --model NN`

##### test model

`python siamese.py --test --model NN`

#### Siamese-CNN model

This model is a simple complementation of a Siamese CNN QA model with a pointwise way.

[To this repo](./siamese_cnn)

##### train model

`python siamese.py --train --model CNN`

##### test model

`python siamese.py --test --model CNN`

#### Siamese-RNN model

This model is a simple complementation of a Siamese RNN/LSTM/GRU QA model with a pointwise way.

[To this repo](./siamese_rnn)

##### train model

`python siamese.py --train --model RNN`

##### test model

`python siamese.py --test --model RNN`

#### note

All these three models above are based on the vanilla siamese structure. You can easily combine these basic deep learning module cells together and build your own models.


### Pairwise Style

#### QACNN

Given a question, a positive answer and a negative answer, this pairwise model can rank two answers with higher ranking in terms of the right answer.

Refer to 《APPLYING DEEP LEARNING TO ANSWER SELECTION:A STUDY AND AN OPEN TASK》

[To this repo](./qacnn)

##### train model

`python qacnn.py --train`

##### test model

`python qacnn.py --test`

### Listwise Style(also can be transformed to pointwise style)

#### Decomposable Attention Model

Refer to 《A Decomposable Attention Model for Natural Language Inference》

[To this repo](./decomposable_att_model)

##### train model

`python decomp_att.py --train`

##### test model

`python decomp_att.py --test`

#### Compare-Aggregate Model with Multi-Compare

Refer to 《A COMPARE-AGGREGATE MODEL FOR MATCHING TEXT SEQUENCES》

[To this repo](./seq_match_seq)

##### train model

`python seq_match_seq.py --train`

##### test model

`python seq_match_seq.py --test`

#### BiMPM

Refer to 《Bilateral Multi-Perspective Matching for Natural Language Sentence》

[To this repo](./bimpm)

##### train model

`python bimpm.py --train`

##### test model

`python bimpm.py --test`

## Machine Reading Comprehension

### Cloze Style

#### Dataset

CNN/Daily mail, CBT

#### GA Reader

To be done

#### SA Reader

To be done

#### AoA Reader

To be done

### Answer Extraction Style

#### Dataset

SQuAD, MS MARCO

#### BiDAF

To be done

#### QANet

Refer to 《QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension》

[To this repo](./QANet)

### Answer Selection Style

#### Dataset

RACE dataset

## Information

For more information, please visit http://skyhigh233.com/blog/2018/04/26/cqa-intro/.
