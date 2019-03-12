'''
Final CNN Model (Has feature for testing on test dataset)
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
from keras.models import load_model


def retrieve_data():
        train_ds = pd.read_csv('CNN_model/data/train_synthetic.csv', sep=',', encoding = "ISO-8859-1")

        X_train = list(train_ds['question'].values)
        y_train = list(train_ds['operation'].values)

        # test_ds = pd.read_csv('CNN_model/data/MAWPS/MAWPS.csv', sep=',', encoding = "ISO-8859-1")
        test_ds = pd.read_csv('CNN_model/data/combination/combined.csv', sep=',', encoding = "ISO-8859-1")

        # test_ds = pd.read_csv('CNN_model/data/reformatted_iit.csv', sep=',', encoding = "ISO-8859-1")

        X_test = list(test_ds['question'].values)
        y_test = list(test_ds['operation'].values)

        return X_train, y_train, X_test, y_test


def retrieve_one_hot_embeddings(questions, word_to_idx):
    embeddings = []
    for qstn in questions:
        embeddings.append([word_to_idx[word] for word in str(qstn).split() if word in word_to_idx])
    return embeddings


def CNNLOG(X_test, y_test, qstn=None, outputDF=False):
    # Create a vocab (word to index)
    vocab_dict_f = open('CNN_model/vocab_dict.pickle', 'rb')
    vocab_dict = pickle.load(vocab_dict_f)
    vocab_dict_f.close()

    vocabulary_size = len(vocab_dict)+1

    X_test_embedding = retrieve_one_hot_embeddings(X_test,vocab_dict)

    # Number of labels would be just 4: +,-,*,/
    num_of_labels = 4
    labels = ['Addition', 'Division', 'Multiplication', 'Subtraction']

    # Create dict for labels to index
    label_to_index = {o:i for i,o in enumerate(labels)}
    index_to_label = {i:o for i,o in enumerate(labels)}


    # Convert labels in the training and test datset to numeric format
    y_test_label_numeric_rep = [label_to_index[label] for label in y_test]

    y_test_distribution = np_utils.to_categorical(y_test_label_numeric_rep, num_of_labels)


    # pad (post) questions to max length
    max_length = 100
    X_test_embedding_padded = pad_sequences(X_test_embedding, maxlen=max_length, padding='post')


    model = load_model('CNN_model/cnn_model_compressed.hdf5')

    y_predict = model.predict(X_test_embedding_padded)
    print(y_predict[:5])
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


def main():
    X_train, y_train, X_test, y_test = retrieve_data()

    # input_question = input('Enter your question? ')
    # quantities = re.findall(r'\d+', input_question)

    y_predict = CNNLOG(X_test, y_test)[0]

    # quantities = list(map(int, quantities))

    # print(y_predict)
    # result = None
    # if y_predict == 'Addition':
    #     result = quantities[0] + quantities[1]
    # elif y_predict == 'Multiplication':
    #     result = quantities[0] * quantities[1]
    # elif y_predict == 'Division':
    #     result = max(quantities) / min(quantities)
    # else:
    #     result = max(quantities) - min(quantities)
    #
    # print('Result: {}'.format(result))


main()
