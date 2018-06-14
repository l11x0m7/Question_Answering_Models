#!/bin/bash


echo "train model"
python decomp_att.py --train


echo "test model"
python decomp_att.py --test
