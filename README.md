# Question_Answering_Models
This repo collects and re-produces models related to domains of question answering and machine reading comprehension

## comunity QA

#### Dataset

WikiQA, TrecQA, InsuranceQA

### Pointwise Style

#### Siamese-NN model

This model is a simple complementation of a Siamese NN QA model with a pointwise way.

##### train model

`python siamese.py --train --model NN`

##### test model

`python siamese.py` --test --model NN`

#### Siamese-CNN model

This model is a simple complementation of a Siamese CNN QA model with a pointwise way.

##### train model

`python siamese.py --train --model CNN`

##### test model

`python siamese.py` --test --model CNN`

#### Siamese-RNN model

This model is a simple complementation of a Siamese RNN/LSTM/GRU QA model with a pointwise way.

##### train model

`python siamese.py --train --model RNN`

##### test model

`python siamese.py` --test --model RNN`

#### note

All these three models above are based on the vanilla siamese structure. You can easily combine these basic deep learning module cells together and build your own models.


### Pairwise Style

#### QACNN

Given a question, a positive answer and a negative answer, this pairwise model can rank two answers with higher ranking in terms of the right answer.

##### train model

`python qacnn.py --train`

##### test model

`python qacnn.py` --test`

### Listwise Style

#### Compare-Aggregate model

To be done

## Machine Reading Comprehension

### Cloze Style

#### Dataset

CNN/Daily mail, CBT

#### GA Reader


#### SA Reader


#### AoA Reader


### Answer Extraction Style

#### Dataset

SQuAD, MS MARCO

#### BiDAF


### Answer Selection Style

#### Dataset

RACE dataset


## Information

For more information, please visit http://skyhigh233.com/blog/2018/04/26/cqa-intro/.