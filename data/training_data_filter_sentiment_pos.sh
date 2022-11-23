#!/bin/sh
'''
python create_training_data_sentiment_pos.py \
    --rates 1  \
    --rates 5 \
    --rates 10 \
    --rates 20 \
    --rates 30 \
    --rates 50
'''
#python create_training_data_sentiment_pos.py \
#    --nums 10000  \
#    --nums 25000 

python create_training_data_sentiment_pos.py \
    --rates 10 \
    --rates 20 \
    --rates 50 \
    --rates 80 \
    --nums 10000 \
    --nums 20000 \
    --nums 25000 \
    --nums 50000 \
    --nums 100000 \
    --nums 125000 \
    --nums 250000
#python -m pdb create_training_data_sentiment_pos.py \
#    --rates 10 