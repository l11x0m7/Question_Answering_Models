#!/bin/bash


echo "train model"
python siamese_cnn.py --train


echo "test model"
python siamese_cnn.py --test
