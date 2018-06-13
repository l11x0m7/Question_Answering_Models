#!/bin/bash


echo "train model"
python siamese_nn.py --train


echo "test model"
python siamese_nn.py --test