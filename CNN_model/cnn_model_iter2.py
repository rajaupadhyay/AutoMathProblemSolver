'''
Second iteration of the CNN model to predict the operation required for a question
Accuracy: 60%
'''
import pandas as pd
import random
from collections import OrderedDict
import numpy as np
import pickle

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras import layers
from keras.optimizers import Adam
import re


'''
Read dataset and split into train+test
'''
data_set = None
# Replaced has questions with 'num' as a placeholder rather
# than having the actual quantities in the questions
with open('data/singleop_shuffled_num_replaced.pickle', 'rb') as handle:
    data_set = pickle.load(handle)

train_size = int(len(data_set) * .7)


X_train = [itx[0] for itx in data_set[:train_size]]
y_train = [itx[1] for itx in data_set[:train_size]]

X_test = [itx[0] for itx in data_set[train_size:]]
y_test = [itx[1] for itx in data_set[train_size:]]



'''
Basic preprocessing
'''
max_words = 200
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(X_train) # only fit on train

X_train_one_hot = tokenize.texts_to_matrix(X_train)
X_test_one_hot = tokenize.texts_to_matrix(X_test)

labels_set = list(set(y_train))
num_classes = len(labels_set)
label_to_index = {o:i for i,o in enumerate(labels_set)}
index_to_label = {i:l for i,l in enumerate(labels_set)}

# Convert labels to numeric format
y_train_label_numeric_rep = [label_to_index[label] for label in y_train]
y_test_label_numeric_rep = [label_to_index[label] for label in y_test]

# Just creates the actual one hot encoded vectors
y_train = utils.to_categorical(y_train_label_numeric_rep, num_classes)
y_test = utils.to_categorical(y_test_label_numeric_rep, num_classes)

batch_size = 32
epochs = 2



def build_vocab(questions):
    # Build vocab and word index
    vocab_set = set()
    for qstn in questions:
        vocab_set = vocab_set.union(set(qstn.split()))
    vocab = list(vocab_set)

    # 0 is used for padding
    word_to_idx = OrderedDict([(w,i) for i,w in enumerate(vocab,1)])

    return word_to_idx


def simple_CNN(X_train, y_train, X_test, y_test, qstn=None):
    vocab_dict = build_vocab(X_train)

    vocabulary_size = len(vocab_dict)+1

    embedding_dim = 256

    model = Sequential()

    # Everything is the same as iteration 1 but lets try adding more convolutional layers
    # and also maxpooling after every conv layer before doing a global max pool after
    # the final layer
    # Remember global max pool will find the max over the entire output whereas
    # maxpool will do it over a specified windows
    # 0,1,2,2,5,1,2 , global max pooling outputs 5, whereas ordinary max pooling
    # layer with pool size equals to 3 outputs 2,2,5,5,5

    model.add(layers.Embedding(vocabulary_size, embedding_dim, input_length=max_words))
    model.add(layers.Conv1D(64, 5, padding='valid', kernel_initializer='normal', activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(128, 5, padding='valid', kernel_initializer='normal', activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(128, 5, padding='valid', kernel_initializer='normal', activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    # model.add(layers.Dense(10, activation='relu'))

    model.add(Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(X_train_one_hot, y_train,
                    epochs=10,
                    verbose=False,
                    validation_split=0.1,
                    batch_size=32)

    score = model.evaluate(X_test_one_hot, y_test,
                           batch_size=batch_size, verbose=0)

    print('Test accuracy:', score[1])

    if qstn:
        qstn_encoded = tokenize.texts_to_matrix([qstn])
        res = model.predict(qstn_encoded)
        y_predict = index_to_label[np.argmax(res)]
        return y_predict



# simple_CNN(X_train, y_train, X_test, y_test, qstn='Tom has 3 apples. Jack gave him 2 more apples. How many apples does Tom have now?')


input_question = input('Enter your question? ')
# input_model_type = input('Vanilla Model (0) or CNN Model (1)? ')
operator_to_use = None

quantities = re.findall(r'\d+', input_question)

for qtn in quantities:
    input_question = input_question.replace(qtn, 'num')


operator_to_use = simple_CNN(X_train, y_train, X_test, y_test, qstn=input_question)

quantities = list(map(int, quantities))

result = None
if operator_to_use == 'Addition':
    result = quantities[0] + quantities[1]
elif operator_to_use == 'Multiplication':
    result = quantities[0] * quantities[1]
elif operator_to_use == 'Division':
    result = max(quantities) / min(quantities)
else:
    result = max(quantities) - min(quantities)

print('Result: {}'.format(result))
