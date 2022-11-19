#!/bin/sh
python create_training_data_sentiment_pos.py --rate 1
python create_training_data_sentiment_pos.py --rate 5
python create_training_data_sentiment_pos.py --rate 10
python create_training_data_sentiment_pos.py --rate 20
python create_training_data_sentiment_pos.py --rate 30
python create_training_data_sentiment_pos.py --rate 50