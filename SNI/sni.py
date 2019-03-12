from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
import numpy as np
from keras.utils import np_utils
import random
from collections import OrderedDict
import re
from keras.callbacks import ModelCheckpoint
import pickle
from keras.models import load_model


def retrieve_one_hot_embeddings(questions, word_to_idx):
    embeddings = []
    for qstn in questions:
        embeddings.append([word_to_idx[word] for word in qstn.split() if word in word_to_idx])
    return embeddings


def SNI_model(input_phrase):
    vocab_dict_f = open('SNI/vocab_dict_for_sni.pickle', 'rb')
    vocab_dict = pickle.load(vocab_dict_f)
    vocab_dict_f.close()

    input_phrase_embedding = retrieve_one_hot_embeddings(input_phrase,vocab_dict)

    # pad (post) questions to max length
    max_length = 10
    input_phrase_embedding = pad_sequences(input_phrase_embedding, maxlen=max_length, padding='post')


    model = load_model('SNI/SNI_model.hdf5')

    sig_prediction = model.predict(input_phrase_embedding)
    sig_prediction = sum(sig_prediction)/len(sig_prediction)

    return sig_prediction


def SNI(question):
    question = question.splitlines()
    question = ' '.join(question)
    quantities = re.findall(r"\d+(?:\.\d+)?", question)
    question_tokens = question.split()
    total_len = len(question_tokens)

    phrase_list = []
    quantities_extracted = []
    indices_to_change = []

    for qnty in quantities:
        if not qnty:
            continue

        idx = None
        try:
            idx = question_tokens.index(qnty)
        except:
            for tkn in range(len(question_tokens)):
                if qnty in question_tokens[tkn]:
                    idx = tkn
                    break

        indices_to_change.append(idx)
        quantities_extracted.append(qnty)
        # get the words around the number (window of size 3)
        left_index = min(filter(lambda x: x>=0, [idx, idx-1, idx-2, idx-3]))
        right_index = max(filter(lambda x: x<total_len, [idx, idx+1, idx+2, idx+3]))
        window_string = ' '.join(question_tokens[left_index:idx]) + ' ' + ' '.join(question_tokens[idx+1:right_index+1])

        phrase_list.append(window_string)

    for phrase in range(len(phrase_list)):
        sig = SNI_model(phrase_list[phrase])
        if sig < 0:
            idx_change = indices_to_change[phrase]
            question_tokens[idx_change] = 'num'

    updated_qstn = ' '.join(question_tokens)
    return updated_qstn

print(SNI("What is the sum of the 2 numbers, 4 and 3?"))
