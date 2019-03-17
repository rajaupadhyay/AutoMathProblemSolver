'''
Final CNN Model
'''
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import pickle
from keras.models import load_model
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def retrieve_one_hot_embeddings(questions, word_to_idx):
    embeddings = []
    for qstn in questions:
        embeddings.append([word_to_idx[word] for word in qstn.split() if word in word_to_idx])
    return embeddings


def CNN_MODEL(qstn):
    vocab_dict_f = open('vocab_dict.pickle', 'rb')
    vocab_dict = pickle.load(vocab_dict_f)
    vocab_dict_f.close()

    vocabulary_size = len(vocab_dict)+1

    # Number of labels would be just 4: +,-,*,/
    num_of_labels = 4
    labels = ['Addition', 'Division', 'Multiplication', 'Subtraction']

    # Create dict for labels to index
    label_to_index = {o:i for i,o in enumerate(labels)}
    index_to_label = {i:o for i,o in enumerate(labels)}

    max_length = 100
    model = load_model('cnn_model_compressed.hdf5')

    qstn_embedding = retrieve_one_hot_embeddings(qstn, vocab_dict)
    qstn_embedding_padded = pad_sequences(qstn_embedding, maxlen=max_length, padding='post')

    y_predict = model.predict(qstn_embedding_padded)
    # Arg max to get the predicted operator
    y_predict = [index_to_label[np.argmax(i)] for i in y_predict]

    return y_predict


def calculate_result(operator, quantities):
    result = None
    if operator == 'Addition':
        result = quantities[0] + quantities[1]
    elif operator == 'Multiplication':
        result = quantities[0] * quantities[1]
    elif operator == 'Division':
        result = max(quantities) / min(quantities)
    else:
        result = max(quantities) - min(quantities)

    return result



def user_input():
    input_question = input('Enter your question? ')
    quantities = re.findall(r"\d+(?:\.\d+)?", input_question)

    predicted_operator = CNN_MODEL([input_question])[0]

    quantities = list(map(float, quantities))

    result = calculate_result(predicted_operator, quantities)

    print('Result: {}'.format(result))


def CNN(input_question, return_operator):
    quantities = re.findall(r"\d+(?:\.\d+)?", input_question)

    predicted_operator = CNN_MODEL([input_question])[0]

    quantities = list(map(float, quantities))

    result = calculate_result(predicted_operator, quantities)

    if return_operator:
        return result, predicted_operator
    else:
        return result


# user_input()
