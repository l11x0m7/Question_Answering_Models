#!/bin/bash


echo "train model"
python siamese_rnn.py --train


echo "test model"
python siamese_rnn.py --test
