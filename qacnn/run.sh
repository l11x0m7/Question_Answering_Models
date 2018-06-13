#!/bin/bash


echo "train model"
python qacnn.py --train


echo "test model"
python qacnn.py --test
