import pandas as pd
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.utils import np_utils
import random
from collections import OrderedDict
import numpy as np
import re
import pickle
from keras.models import load_model
from reformat_data import obtain_train_and_test
from SVM import train
from tfidf import *

def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

## CNN HELPER FUNCTIONS
def retrieve_data():
    # train_ds, test_ds, train_SVM_data, test_SVM_data = obtain_train_and_test()
    # tfidfX_train, tfidfX_test, tfidfy_train, tfidfy_test, training_ds, testing_ds = obtain_train_and_test()


    return X_train, y_train, X_test, y_test, train_SVM_data, test_SVM_data


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


def train_CNN(X_train, y_train, X_test, y_test, qstn=None, outputDF=False):
    # Create a vocab (word to index)
    vocab_dict = build_vocab(X_train)

    pickle_object(vocab_dict, 'vocab_dict_comb_test')

    vocabulary_size = len(vocab_dict)+1

    # Create one-hot-vector encodings
    # These are not really one-hot-vectors in this file
    # Its just a vector of the word indices for each word in the question
    # Need to try one-hot-vecs also
    X_train_embedding = retrieve_one_hot_embeddings(X_train,vocab_dict)
    X_test_embedding = retrieve_one_hot_embeddings(X_test,vocab_dict)

    # Number of labels would be just 4: +,-,*,/
    num_of_labels = len(set(y_train))
    labels = list(set(y_train))

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

    embedding_dim = 256
    filter_sizes = [3, 4, 5]
    num_filters = 128
    std_drop = 0.5

    epochs = 15
    # batch_size = 128
    batch_size = 256


    print("Creating CNN Model...")
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

    checkpoint = ModelCheckpoint('trained_CNN_model.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    # adam = Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    print("Training CNN Model...")
    # model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[checkpoint],
    #      validation_data=(X_dev_onehot, y_dev_distribution))

    model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[checkpoint],
         validation_data=(X_dev_onehot, y_dev_distribution))




def train_all_models():


    tfidfX_train, tfidfX_test, tfidfy_train, tfidfy_test, training_ds, testing_ds = obtain_train_and_test()

    # TRAIN SVM
    print('Training SVM')
    svmFitFunction, svmFitFunctionWithAdvice = train(training_ds)

    # # TRAIN CNN MODEL
    # train_CNN(X_train, y_train, X_test, y_test)

    tfidf_mdl = TFIDF(0.6, 5, 25)
    tfidf_mdl.fit(tfidfX_train, tfidfy_train)


    return svmFitFunction, tfidf_mdl, testing_ds


# train_all_models()
