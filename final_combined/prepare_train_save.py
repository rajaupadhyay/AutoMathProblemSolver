import pandas as pd
import json
import re
from random import shuffle
import numpy as np
from helper_functions import *

json_object_1_f = open('COMBINED_MODEL/data.json')
dolphin18K_ds = json.load(json_object_1_f)
json_object_1_f.close()

# dolphin18K_ds = [x for x in dolphin18K_ds if x['noEquations'] == 1]

# 14104 questions
TOTAL_QUESTIONS = len(dolphin18K_ds)

# Shuffle and split the dataset
shuffle(dolphin18K_ds)

# Training: 75%, Testing 25%
training_ds = dolphin18K_ds[:int(0.75*TOTAL_QUESTIONS)]
testing_ds = dolphin18K_ds[int(0.75*TOTAL_QUESTIONS):]



def get_corpus(data):
    corpus = []
    for problem in data:
        corpus.append(problem['question'])
    return corpus


def prep_equation_list(qstns, equations):
    equations_list = []
    answers_list = []


    for i in range(len(equations)):
        words = qstns[i].split(' ')
        words = removeEmptiesAndPunctuation(words)
        wordsAndEquations = replaceNumbers(words, [equations[i]], ['X'])

        words = wordsAndEquations[0]
        eqTemplates = wordsAndEquations[1]
        # print(eqTemplates)
        equations_list.append(eqTemplates)
        # answers_list.append(datapoint['answers'])
    return equations_list, answers_list


def prep_answers(data):
    answers_list = []


def createTfidfTrainAndTest(training_ds):
    fp = open('data/equations.json')
    equation_dict = json.load(fp)

    # Preparing Corpus
    # corpus = get_corpus(training_ds)
    corpus = list(training_ds['question'].values)


    equations = list(training_ds['equation'].values)

    # Preparing Equations
    equations_list, _ = prep_equation_list(corpus, equations)

    eqns_in_dict = [i for i, x in enumerate(equations_list) if x in equation_dict.keys()]
    corpus = [corpus[i] for i in eqns_in_dict]
    equations_list = [equations_list[i] for i in eqns_in_dict]

    equation_values = list()
    for i in range(len(equations_list)):
        equation_values.append(equation_dict[equations_list[i]])

    X_test = get_corpus(testing_ds)
    # Y_test, answers_list = prep_equation_list(training_ds)
    Y_test = None
    return corpus, X_test, equation_values, Y_test



def obtain_train_and_test():
    train_synthetic = pd.read_csv('data/train_synthetic.csv', sep=',')
    tfidfX_train, tfidfX_test, tfidfy_train, tfidfy_test = createTfidfTrainAndTest(train_synthetic, testing_ds)


    return tfidfX_train, tfidfX_test, tfidfy_train, tfidfy_test, training_ds, testing_ds
