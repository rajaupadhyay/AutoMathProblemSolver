'''
Third iteration of the CNN model to predict the operation required for a question
Accuracy: 83% (Without)
'''
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

def retrieve_data(flag=0):
    if flag == 0:
        train_ds = pd.read_csv('data/new_train.csv', sep=',', encoding = "ISO-8859-1")
        test_ds = pd.read_csv('data/test.csv', sep=',')

        X_train = list(train_ds['question'].values)
        y_train = list(train_ds['operation'].values)

        X_test = list(test_ds['question'].values)
        y_test = list(test_ds['operation'].values)

        return X_train, y_train, X_test, y_test
    elif flag == 1:
        df = pd.read_csv('data/formatted_singleop.csv', sep=',', encoding = "ISO-8859-1")
        train_size = int(len(df) * .8)

        questions = list(df['question'].values)
        ops = list(df['operation'].values)

        combined = list(zip(questions, ops))
        random.shuffle(combined)

        X_train = [itx[0] for itx in combined[:train_size]]
        y_train = [itx[1] for itx in combined[:train_size]]

        X_test = [itx[0] for itx in combined[train_size:]]
        y_test = [itx[1] for itx in combined[train_size:]]

        return X_train, y_train, X_test, y_test
    elif flag == 2:
        data_set = None
        with open('data/singleop_shuffled_num_replaced.pickle', 'rb') as handle:
            data_set = pickle.load(handle)

        train_size = int(len(data_set) * .7)

        X_train = [itx[0] for itx in data_set[:train_size]]
        y_train = [itx[1] for itx in data_set[:train_size]]

        X_test = [itx[0] for itx in data_set[train_size:]]
        y_test = [itx[1] for itx in data_set[train_size:]]

        return X_train, y_train, X_test, y_test


def shuffle_list(*ls):
    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

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


def CNNLOG(X_train, y_train, X_test, y_test, qstn=None):
    # Create a vocab (word to index)
    vocab_dict = build_vocab(X_train)

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

    # These values provide the best perfomance
    embedding_dim = 256
    filter_sizes = [3, 4, 5]
    num_filters = 128
    std_drop = 0.2

    epochs = 10
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

    # checkpoint = ModelCheckpoint('CNN_iter3.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    adam = Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    print("Training Model...")
    # model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[checkpoint],
    #      validation_data=(X_dev_onehot, y_dev_distribution))

    model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=0,
         validation_data=(X_dev_onehot, y_dev_distribution))

    # model.load_weights("CNN.hdf5")
    y_predict = model.predict(X_test_embedding_padded)
    # Arg max to get the predicted operator
    y_predict = [index_to_label[np.argmax(i)] for i in y_predict]
    # print(y_predict)
    print("Operation acc: {}".format(model.evaluate(X_test_embedding_padded,y_test_distribution,verbose=0)[1]))

    if qstn:
        qstn_embedding = retrieve_one_hot_embeddings(qstn, vocab_dict)
        qstn_embedding_padded = pad_sequences(qstn_embedding, maxlen=max_length, padding='post')

        y_predict = model.predict(qstn_embedding_padded)
        # Arg max to get the predicted operator
        y_predict = [index_to_label[np.argmax(i)] for i in y_predict]



    return y_predict



def main():
    flag = 2
    X_train, y_train, X_test, y_test = retrieve_data(flag=flag)

    input_question = input('Enter your question? ')
    quantities = re.findall(r'\d+', input_question)

    if flag == 2:
        for qtn in quantities:
            input_question = input_question.replace(qtn, 'num')


    y_predict = CNNLOG(X_train, y_train, X_test, y_test, [input_question])[0]

    quantities = list(map(int, quantities))

    result = None
    if y_predict == 'Addition':
        result = quantities[0] + quantities[1]
    elif y_predict == 'Multiplication':
        result = quantities[0] * quantities[1]
    elif y_predict == 'Division':
        result = max(quantities) / min(quantities)
    else:
        result = max(quantities) - min(quantities)

    print('Result: {}'.format(result))


main()
