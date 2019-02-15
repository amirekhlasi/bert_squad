#!/bin/bash

python HP.py
python bert_prepare.py
python data-pre.py
python create_data.py