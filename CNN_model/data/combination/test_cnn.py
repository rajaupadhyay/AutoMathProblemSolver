'''
Testing the CNN model on the Dolphin dataset (12% Accuracy)
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
import json


def retrieve_data(tfidfDataFile='0.5'):
        train_ds = pd.read_csv('CNN_model/data/train_synthetic.csv', sep=',', encoding = "ISO-8859-1")

        X_train = list(train_ds['question'].values)
        y_train = list(train_ds['operation'].values)

        # test_ds = pd.read_csv('CNN_model/data/MAWPS/MAWPS.csv', sep=',', encoding = "ISO-8859-1")
        # test_ds = pd.read_csv('CNN_model/data/combination/combined.csv', sep=',', encoding = "ISO-8859-1")
        test_ds = pd.read_csv('tfidfstuff/dataTFIDF/tfidf_{}.csv'.format(tfidfDataFile), sep=',', encoding = "ISO-8859-1")

        # test_ds = pd.read_csv('CNN_model/data/reformatted_iit.csv', sep=',', encoding = "ISO-8859-1")

        X_test = list(test_ds['question'].values)
        y_test = list(test_ds['operation'].values)

        return X_train, y_train, X_test, y_test


def retrieve_one_hot_embeddings(questions, word_to_idx):
    embeddings = []
    for qstn in questions:
        embeddings.append([word_to_idx[word] for word in str(qstn).split() if word in word_to_idx])
    return embeddings


def CNNLOG(X_test, y_test, qstn=None, outputDF=False, confidenceThreshold=0.5, outputIncorrectQuestionsBelowThreshold=False, TFIDF_SIM='0.5'):
    # Create a vocab (word to index)
    vocab_dict_f = open('CNN_model/vocab_dict.pickle', 'rb')
    vocab_dict = pickle.load(vocab_dict_f)
    vocab_dict_f.close()



    # test_ds = pd.read_csv('tfidfstuff/dataTFIDF/tfidf_{}.csv'.format(TFIDF_SIM), sep=',', encoding = "ISO-8859-1")
    #
    # # test_ds = pd.read_csv('CNN_model/data/reformatted_iit.csv', sep=',', encoding = "ISO-8859-1")
    #
    # X_test = list(test_ds['question'].values)
    # y_test = list(test_ds['operation'].values)
    # noUnknowns = list(test_ds['noUnknowns'].values)
    # unknowns = list(test_ds['unknowns'].values)
    # noEquations = list(test_ds['noEquations'].values)
    # equations = list(test_ds['equations'].values)
    # answers = list(test_ds['answers'].values)





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


    #########################################################
    predictions = []
    relevantIndices = []
    relevantIndicesForQuestionsBelowThreshold = []

    for preds in range(len(y_predict)):
        maxPredVal = max(y_predict[preds])
        if maxPredVal >= confidenceThreshold:
            predictions.append(index_to_label[np.argmax(y_predict[preds])])
            relevantIndices.append(preds)
        else:
            relevantIndicesForQuestionsBelowThreshold.append(preds)

    goldLabels = [index_to_label[np.argmax(y_test_distribution[i])] for i in relevantIndices]
    print('##################{}####################'.format(confidenceThreshold))
    print('Confidence Threshold: ', confidenceThreshold)
    print('Total questions: ', len(y_predict))
    print('Questions below the threshold', len(relevantIndicesForQuestionsBelowThreshold))
    correctValues = sum([predictions[x] == goldLabels[x] for x in range(len(predictions))])

    acc = correctValues/len(predictions)
    print('Accuracy', acc)
    print('Correct questions: ', acc * len(predictions))
    print('######################################')


    if outputIncorrectQuestionsBelowThreshold:
        output_df = pd.DataFrame()
        test_qs = [X_test[itx] for itx in relevantIndicesForQuestionsBelowThreshold if isinstance(X_test[itx], str)]
        test_ops = [y_test[itx] for itx in relevantIndicesForQuestionsBelowThreshold]

        test_noUnknowns = [noUnknowns[itx] for itx in relevantIndicesForQuestionsBelowThreshold]
        test_unknowns = [[unknowns[itx]] for itx in relevantIndicesForQuestionsBelowThreshold]
        test_noEquations = [noEquations[itx] for itx in relevantIndicesForQuestionsBelowThreshold]
        test_equations = [[equations[itx]] for itx in relevantIndicesForQuestionsBelowThreshold]
        test_answers = [[[answers[itx]]] for itx in relevantIndicesForQuestionsBelowThreshold]


        output_df['question'] = pd.Series(test_qs)
        output_df['operation'] = pd.Series(test_ops)

        output_df['noUnknowns'] = pd.Series(test_noUnknowns)
        output_df['unknowns'] = pd.Series(test_unknowns)
        output_df['noEquations'] = pd.Series(test_noEquations)
        output_df['equations'] = pd.Series(test_equations)
        output_df['answers'] = pd.Series(test_answers)

        out_json = output_df.to_json('QBT{}_tfidfSim{}.json'.format(confidenceThreshold, TFIDF_SIM), orient='records')

        # with open('QBT{}_tfidfSim{}.json'.format(confidenceThreshold, TFIDF_SIM), 'w') as f:
        #     f.write(out_json)


        # output_df.to_csv('QBT{}_tfidfSim{}.csv'.format(confidenceThreshold, TFIDF_SIM), sep=',')



    ###########################################################

    # Arg max to get the predicted operator
    y_predict = [index_to_label[np.argmax(i)] for i in y_predict]
    # print(y_predict)
    # print("Operation acc: {}".format(model.evaluate(X_test_embedding_padded,y_test_distribution,verbose=0)[1]))

    if outputDF:
        output_df = pd.DataFrame()
        output_df['question'] = pd.Series(X_test)
        output_df['operator'] = pd.Series(y_test)
        output_df['predicted_operator'] = pd.Series(y_predict)
        output_df = output_df[output_df['operator'] != output_df['predicted_operator']]

        output_df.to_csv('CNN_model_iter3_predictions.csv', sep=',')



    if qstn:
        qstn_embedding = retrieve_one_hot_embeddings(qstn, vocab_dict)
        qstn_embedding_padded = pad_sequences(qstn_embedding, maxlen=max_length, padding='post')

        y_predict = model.predict(qstn_embedding_padded)
        # Arg max to get the predicted operator
        y_predict = [index_to_label[np.argmax(i)] for i in y_predict]

    return y_predict


def main():
    tfidf_sim_vals = [0.9]
    # X_train, y_train, X_test, y_test = retrieve_data(retrieve_data='0.5')
    #
    # # y_predict = CNNLOG(X_test, y_test, confidenceThreshold=0.4)[0]
    # # y_predict = CNNLOG(X_test, y_test, confidenceThreshold=0.5)[0]
    # y_predict = CNNLOG(X_test, y_test, confidenceThreshold=0.6, outputIncorrectQuestionsBelowThreshold=True, TFIDF_SIM='0.5')[0]
    # # y_predict = CNNLOG(X_test, y_test)[0]

    for tfifVal in tfidf_sim_vals:
        X_train, y_train, X_test, y_test = retrieve_data(tfidfDataFile=str(tfifVal))
        y_predict = CNNLOG(X_test, y_test, confidenceThreshold=0.0)[0]




main()
