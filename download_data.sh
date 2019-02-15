#!/bin/bash

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O BERT.zip
unzip BERT.zip
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -P data/
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -P data/