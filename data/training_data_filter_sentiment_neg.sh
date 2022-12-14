#!/bin/sh
'''
python create_training_data_sentiment_neg.py \
    --rates 1  \
    --rates 5 \
    --rates 10 \
    --rates 20 \
    --rates 30 \
    --rates 50
'''
python create_training_data_sentiment_neg.py \
    --rates 10 \
    --rates 20 \
    --rates 50 \
    --rates 80 \
    --nums 125000 

python create_training_data_sentiment_neg.py \
    --nums 10000  \
    --nums 25000 

#python create_training_data_sentiment_neg.py --rate 5
#python create_training_data_sentiment_neg.py --rate 10
#python create_training_data_sentiment_neg.py --rate 20
#python create_training_data_sentiment_neg.py --rate 30
#python create_training_data_sentiment_neg.py --rate 50