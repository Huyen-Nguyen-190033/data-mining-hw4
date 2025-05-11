
import torch
from torch import nn
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter


'''step 1: read the raw data'''
def _read_data():
	##-------------------------------------------
	## read the training and test data from files
	##-------------------------------------------
	train_data = pd.read_csv("/Users/huyennguyen/Downloads/homework4/training_raw_data.csv")
	test_data = pd.read_csv("/Users/huyennguyen/Downloads/homework4/test_raw_data.csv")
	return train_data, test_data


'''step 2: data preprocessing'''
def _data_preprocessing(input_data):
    input_data['clean_text'] = input_data['Content'].map(lambda x: re.sub(r'[^\w\s]', ' ', x))

    ## convert words to lower case
    input_data['clean_text'] = input_data['clean_text'].str.lower()

    return input_data


'''step 3: stats of length of sentences'''
def _stats_seq_len(input_data):
    input_data['seq_words'] = input_data['clean_text'].apply(lambda x: x.split())
    input_data['seq_len'] = input_data['seq_words'].apply(lambda x: len(x))
    print(input_data['seq_len'].describe())
    
    ## remove short and long tokens
    min_seq_len = 100
    max_seq_len = 600
    input_data = input_data[min_seq_len <= input_data['seq_len']]
    input_data = input_data[input_data['seq_len'] <= max_seq_len]

    return input_data


'''step 4: convert 'positive and negative' to labels 1 and 0'''
def _convert_labels(input_data):
	## convert "pos" and "neg" to boolean values 1 and 0
	input_data['Label'] = input_data['Label'].map({'pos': 1, 'neg': 0})
	return input_data


'''step 5: Tokenize: create Vocab to Index mapping dictionary'''
def _map_tokens2index(input_data, top_K=500):
    words = input_data['seq_words'].tolist()
    tokens_list = []
    for l in words:
        tokens_list.extend(l)

    ## count the frequency of words
    count_tokens = Counter(tokens_list)

    ## dictionary = {words: count}
    sorted_tokens = count_tokens.most_common(len(tokens_list))

    ## choose the top K tokens or all of them
    tokens_top = sorted_tokens[:top_K]

    ## tokens to index starting from 2, index=0:<padding>, index=1:<unknown>
    tokens2index = {w: i+2 for i, (w, c) in enumerate(tokens_top)}

    ## add index for padding (0) and unknown (1)
    tokens2index['<pad>'] = 0
    tokens2index['<unk>'] = 1

    ## save tokens2index to json file
    with open('tokens2index.json', 'w') as outfile:
        json.dump(tokens2index, outfile, indent=4)

    return tokens2index


'''step 6: Tokenize: Encode the words in sentences to index'''
def _encode_word2index(x, tokens2index):
    ## unknown words: index=1
    input_tokens = [tokens2index.get(w, 1) for w in x]
    return input_tokens


'''step 7: Padding or truncating sequence data'''
def _pad_or_truncate(input_tokens, max_len=100):
    if len(input_tokens) > max_len:
        return input_tokens[:max_len]
    else:
        return input_tokens + [0] * (max_len - len(input_tokens))


if __name__ == "__main__":
    # Read data
    train_data, test_data = _read_data()

    # Preprocess data
    train_data = _data_preprocessing(train_data)
    test_data = _data_preprocessing(test_data)

    # Stats and filtering
    train_data = _stats_seq_len(train_data)
    test_data = _stats_seq_len(test_data)

    # Convert labels
    train_data = _convert_labels(train_data)
    test_data = _convert_labels(test_data)

    # Create vocab mapping
    tokens2index = _map_tokens2index(train_data, top_K=8000)

    # Encode
    train_data['encoded'] = train_data['seq_words'].apply(lambda x: _encode_word2index(x, tokens2index))
    test_data['encoded'] = test_data['seq_words'].apply(lambda x: _encode_word2index(x, tokens2index))

    # Pad/truncate
    train_data['padded'] = train_data['encoded'].apply(lambda x: _pad_or_truncate(x, max_len=100))
    test_data['padded'] = test_data['encoded'].apply(lambda x: _pad_or_truncate(x, max_len=100))

    # Save processed data
    train_data.to_csv("/Users/huyennguyen/Downloads/homework4/training_processed.csv", index=False)
    test_data.to_csv("/Users/huyennguyen/Downloads/homework4/test_processed.csv", index=False)

    print("Data preprocessing complete!")
