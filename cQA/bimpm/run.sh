#!/bin/bash


echo "train model"
python bimpm.py --train


echo "test model"
python bimpm.py --test
