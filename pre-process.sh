#!/bin/bash

python bert_prepare.py
python HP.py
python data-pre.py
python create_data.py
