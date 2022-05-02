#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fashion_captioning
python fashionpedia_caption.py $@ 
