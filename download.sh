#!/usr/bin/env bash

PWD=$(pwd)

# Download GloVe
GLOVE_DIR=$PWD/data/embedding
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.6B.300d.zip -O $GLOVE_DIR/glove.6B.300d.zip
unzip $GLOVE_DIR/glove.6B.300d.zip -d $GLOVE_DIR

# Download Glove Character Embedding
# wget https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt -O $GLOVE_DIR/glove.840B.300d-char.txt

# Download fasttext
# FASTTEXT_DIR=~/data/fasttext
# mkdir -p $FASTTEXT_DIR
# wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip -O $FASTTEXT_DIR/wiki-news-300d-1M.vec.zip
# unzip $FASTTEXT_DIR/wiki-news-300d-1M.vec.zip -d $FASTTEXT_DIR