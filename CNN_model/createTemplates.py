import pandas as pd
from encoder_functions import *
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import json
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.utils import np_utils
import random
import numpy as np
import re
import pickle
from keras.models import load_model
from nltk import pos_tag, word_tokenize
from math_terms import math_terms

def prep_equation_list(filepath):
    equations_list = []

    with open(filepath) as file:
        data = json.load(file)

    for datapoint in data:
        words = datapoint['question'].split(' ')
        words = removeEmptiesAndPunctuation(words)
        wordsAndEquations = replaceNumbers(words, datapoint['equations'], datapoint['unknowns'])

        words = wordsAndEquations[0]
        eqTemplates = wordsAndEquations[1]
        equations_list.append(eqTemplates)
    return equations_list


filepath = 'CNN_model/data2/data.json'


def get_corpus():
    corpus = []
    with open(filepath) as file:
        data = json.load(file)
    for problem in data:
        corpus.append(problem['question'])
    return corpus



fp = open('SVM/data/equations.json')
equation_dict = json.load(fp)

# Preparing Corpus
questionsInCorpus = get_corpus()

# Preparing Equations
equations_list = prep_equation_list(filepath)

eqns_in_dict = [i for i, x in enumerate(equations_list) if x in equation_dict.keys() ]
questionsInCorpus = [questionsInCorpus[i] for i in eqns_in_dict]
equations_list = [equations_list[i] for i in eqns_in_dict]

equation_values = list()
for i in range(len(equations_list)):
  equation_values.append(equation_dict[equations_list[i]])

def replace_nouns(corpus, window_size = 3):
    for j in range(len(corpus)):
        count = window_size + 1
        pos = pos_tag(word_tokenize(corpus[j]))
        for i in range(len(pos)):

            try:
                float(pos[i][0])
                count = 0
                continue
            except ValueError:
                count += 1

            if (pos[i][1] == 'NNP' or pos[i][1] == 'NNPS') and i > 0:
                corpus[j] = str.replace(corpus[j], pos[i][0], 'PN')
                continue

            if count > window_size:
                continue;

            if pos[i][0].lower() in math_terms:
                continue

            if pos[i][1] == 'NN' or pos[i][1] == 'NNS':
                corpus[j] = str.replace(corpus[j], pos[i][0], 'CN')
    return corpus

questionsInCorpus = replace_nouns(questionsInCorpus)



X_train, X_test, y_train, y_test = train_test_split(questionsInCorpus, equation_values, test_size = 0.25, random_state=1)




print('Total qstns', len(questionsInCorpus))




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


def CNN_MODEL(X_train, y_train, X_test, y_test, qstn=None, outputDF=False):
    # Create a vocab (word to index)
    vocab_dict = build_vocab(X_train)

    # pickle_object(vocab_dict, 'vocab_dict')

    vocabulary_size = len(vocab_dict)+1

    total_labels = y_train + y_test


    # Create one-hot-vector encodings
    # These are not really one-hot-vectors in this file
    # Its just a vector of the word indices for each word in the question
    # Need to try one-hot-vecs also
    X_train_embedding = retrieve_one_hot_embeddings(X_train,vocab_dict)
    X_test_embedding = retrieve_one_hot_embeddings(X_test,vocab_dict)

    # Number of labels would be just 4: +,-,*,/
    num_of_labels = len(set(total_labels))
    labels = list(set(total_labels))
    print(labels)

    # Create dict for labels to index
    label_to_index = {o:i for i,o in enumerate(labels)}
    index_to_label = {i:o for i,o in enumerate(labels)}


    # Convert labels in the training and test datset to numeric format
    y_train_label_numeric_rep = [label_to_index[label] for label in y_train]
    y_test_label_numeric_rep = [label_to_index[label] for label in y_test]

    # Just creates the actual one hot encoded vectors
    # e.g. 0 : [0 0 0 0]
    # 1: [0 1 0 0]
    y_train_distribution = np_utils.to_categorical(y_train_label_numeric_rep, num_of_labels)
    y_test_distribution = np_utils.to_categorical(y_test_label_numeric_rep, num_of_labels)


    # pad (post) questions to max length
    max_length = 100
    X_train_embedding_padded = pad_sequences(X_train_embedding, maxlen=max_length, padding='post')
    X_test_embedding_padded = pad_sequences(X_test_embedding, maxlen=max_length, padding='post')

    X_shuffled, y_shuffled = X_train_embedding_padded, y_train_distribution
    length = len(X_shuffled)

    # Split the training dataset into train (80%) + dev (20%)
    X_train_onehot = np.array(X_shuffled[:int(0.8*length)])
    X_dev_onehot = np.array(X_shuffled[int(0.8*length):])
    y_train_distribution = np.array(y_shuffled[:int(0.8*length)])
    y_dev_distribution = np.array(y_shuffled[int(0.8*length):])


    # embedding_dim = 256
    # filter_sizes = [3, 4, 5]
    # num_filters = 128
    # std_drop = 0.2
    #
    # epochs = 10
    # batch_size = 128

    '''
    These settings provide an accuracy of 82.6% on the IIIT test dataset
    The result is better than the result obtained from ARIS.
    '''
    embedding_dim = 256
    filter_sizes = [3, 4, 5]
    num_filters = 128
    std_drop = 0.5

    epochs = 15
    batch_size = 128


    print("Creating Model...")
    inputs = Input(shape=(max_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_length)(inputs)
    reshape = Reshape((max_length, embedding_dim, 1))(embedding)

    # Kernel size specifies the size of the 2-D conv window
    # looking at 3 words at a time in the 1st layer, 4 in the 2nd ...
    # set padding to valid to ensure no padding
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal',
                activation='relu')(reshape)

    # Pool size is the downscaling factor
    maxpool_0 = MaxPool2D(pool_size=(max_length-filter_sizes[0]+1, 1), strides=(2,2), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_length-filter_sizes[1]+1, 1), strides=(2,2), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_length-filter_sizes[2]+1, 1), strides=(2,2), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(std_drop)(flatten)
    output = Dense(units=num_of_labels, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)

    # checkpoint = ModelCheckpoint('model_flag_0.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    # adam = Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    print("Training Model...")
    # model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[checkpoint],
    #      validation_data=(X_dev_onehot, y_dev_distribution))

    model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=1,
         validation_data=(X_dev_onehot, y_dev_distribution))



    # model.save('model_flag_0.h5')
    # model = load_model('model_flag_0.hdf5')


    # model.load_weights("CNN.hdf5")
    y_predict = model.predict(X_test_embedding_padded)
    # Arg max to get the predicted operator
    y_predict = [index_to_label[np.argmax(i)] for i in y_predict]
    # print(y_predict)
    print("Operation acc: {}".format(model.evaluate(X_test_embedding_padded,y_test_distribution,verbose=0)[1]))

    if outputDF:
        output_df = pd.DataFrame()
        output_df['question'] = pd.Series(X_test)
        output_df['operator'] = pd.Series(y_test)
        output_df['predicted_operator'] = pd.Series(y_predict)

        output_df.to_csv('CNN_model_iter3_predictions.csv', sep=',')


    if qstn:
        qstn_embedding = retrieve_one_hot_embeddings(qstn, vocab_dict)
        qstn_embedding_padded = pad_sequences(qstn_embedding, maxlen=max_length, padding='post')

        y_predict = model.predict(qstn_embedding_padded)
        # Arg max to get the predicted operator
        y_predict = [index_to_label[np.argmax(i)] for i in y_predict]

    return y_predict



y_predict = CNN_MODEL(X_train, y_train, X_test, y_test)
