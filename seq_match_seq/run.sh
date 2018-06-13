#!/bin/bash


echo "train model"
python seq_match_seq.py --train


echo "test model"
python seq_match_seq.py --test
