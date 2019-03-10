from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
import numpy as np
from keras.utils import np_utils
import random
from collections import OrderedDict
import re

def load_data():
    print ('Loading data...')
    df = pd.read_csv('../SNI/sni_dataset.csv', sep=',', encoding = "ISO-8859-1")

    df = df.reindex(np.random.permutation(df.index))

    phrases = list(df['phrase'].values)
    sig_labels = list(df['sig_label'].values)

    train_size = int(len(phrases) * .7)

    X_train = phrases[:train_size]
    y_train = sig_labels[:train_size]
    X_test = phrases[train_size:]
    y_test = sig_labels[train_size:]

    return X_train, y_train, X_test, y_test


def build_vocab(questions):
    # Build vocab and word index
    vocab_set = set()
    for qstn in questions:
        vocab_set = vocab_set.union(set(qstn.split()))
    vocab = list(vocab_set)

    # 0 is used for padding
    word_to_idx = OrderedDict([(w,i) for i,w in enumerate(vocab,1)])

    return word_to_idx

def retrieve_one_hot_embeddings(questions, word_to_idx):
    embeddings = []
    for qstn in questions:
        embeddings.append([word_to_idx[word] for word in qstn.split() if word in word_to_idx])
    return embeddings




def model(X_train, y_train, X_test, y_test):
    vocab_dict = build_vocab(X_train)

    vocabulary_size = len(vocab_dict)+1

    # Create one-hot-vector encodings
    # These are not really one-hot-vectors in this file
    # Its just a vector of the word indices for each word in the question
    # Need to try one-hot-vecs also
    X_train_embedding = retrieve_one_hot_embeddings(X_train,vocab_dict)
    X_test_embedding = retrieve_one_hot_embeddings(X_test,vocab_dict)

    # pad (post) questions to max length
    max_length = 10
    X_train_embedding_padded = pad_sequences(X_train_embedding, maxlen=max_length, padding='post')
    X_test_embedding_padded = pad_sequences(X_test_embedding, maxlen=max_length, padding='post')

    X_shuffled, y_shuffled = X_train_embedding_padded, y_train
    length = len(X_shuffled)

    # Split the training dataset into train (80%) + dev (20%)
    X_train_onehot = np.array(X_shuffled[:int(0.8*length)])
    X_dev_onehot = np.array(X_shuffled[int(0.8*length):])
    y_train_distribution = np.array(y_shuffled[:int(0.8*length)])
    y_dev_distribution = np.array(y_shuffled[int(0.8*length):])


    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = vocabulary_size, output_dim = 50, input_length = 10))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=False))
    model.add(Dropout(0.5))
    # model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    hist = model.fit(X_train_onehot, y_train_distribution, batch_size=64, nb_epoch=10, validation_split = 0.1, verbose = 1)

    y_test = np.array(y_test)

    score, acc = model.evaluate(X_test_embedding_padded, y_test, batch_size=1)

    print('Test score:', score)
    print('Test accuracy:', acc)



X_train, y_train, X_test, y_test = load_data()

model = model(X_train, y_train, X_test, y_test)
