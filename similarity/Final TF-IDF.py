#!/usr/bin/env python
# coding: utf-8

# In[18]:


import re
import json
import string
import random
import pandas as pd
from nltk import pos_tag
from random import shuffle
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0, '../SVM')
from encoder_functions import *


# ### POS Tagging
# NN	  noun, singular 'desk'
# NNS	  noun plural	'desks'
# NNP	  proper noun, singular	'Harrison'
# NNPS  proper noun, plural	'Americans'

# In[4]:


# http://lbcrs.org/common/pages/DisplayFile.aspx%3FitemId%3D3446744
# https://www.purplemath.com/modules/translat.htm
MATH_TERMS = [
    # Addition Words
    'add',
    'all',
    'together',
    'altogether',
    'and',
    'both',
    'combined',
    'much',
    'increase',
    'increased',
    'by',
    'plus',
    'sum',
    'total',
    'added',
    'addition',
    # Subtraction words
    'change',
    'decrease',
    'decreased',
    'difference',
    'fewer',
    'left',
    'many',
    'more',
    'longer',
    'shorter',
    'taller',
    'heavier',
    'less',
    'lost',
    'minus',
    'need',
    'reduce',
    'remain',
    'subtract',
    'subtraction',
    'take' ,
    'away',
    'over',
    'after',
    'save',
    'comparative',
    # Multiplication words
    'double',
    'each' ,
    'group',
    'every',
    'factor',
    'multiplied',
    'of',
    'product',
    'times',
    'triple',
    'twice',
    'multiplication',
    'multiply',
    # Division Words
    'cut',
    'share',
    'half',
    'fraction',
    'parts',
    'per',
    'percent',
    'quotient',
    'ratio',
    'separated',
    'equally',
    'divide',
    'division',
    'equal',
    'pieces',
    'split',
    'average',
    # Equality Words
    'is',
    'are',
    'was',
    'were',
    'will',
    'gives',
    'yields',
    'sold',
    'cost',
]


# ## TF-IDF

# In[50]:


class TDIDF:

    def __init__(self, threshold, num_consider):
        self.threshold = threshold
        self.num_consider = num_consider


    def get_tfidf_matrix(self, corpus):
        self.tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        self.tfidf_matrix = self.tf.fit_transform(corpus)


    def replace_nouns(self, corpus):
        for j in range(len(corpus)):
            pos = pos_tag(word_tokenize(corpus[j]))
            for i in range(len(pos)):
                if pos[i][0].lower() in MATH_TERMS:
                    continue
                if pos[i][1] == 'NN' or pos[i][1] == 'NNS':
                    corpus[j] = str.replace(corpus[j], pos[i][0], 'CN')
                elif pos[i][1] == 'NNP' or pos[i][1] == 'NNPS':
                    corpus[j] = str.replace(corpus[j], pos[i][0], 'PN')
        return corpus


    # Find documents similar to a document that is not in the tfidf_matrix
    def find_similar(self, new_doc):
        cosine_similarities = linear_kernel(new_doc[0], self.tfidf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
        return [(index, cosine_similarities[index]) for index in related_docs_indices][0:self.num_consider]


    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.get_tfidf_matrix(X_train)


    def predict(self, X_test):
        y_pred = []
        non_sim_index = []
        for i in range(len(X_test)):
            question = str(X_test[i])
            temp = question.split(" ")
            temp = removeEmptiesAndPunctuation(temp)

            numbers = findNumbersInWords(temp)
            question = self.replace_nouns([question])[0]
            new_doc = self.tf.transform([question])

            template_found = False

            for index, score in self.find_similar(new_doc):
                if score > self.threshold:
                    similar_question = self.X_train[index]
                    similar_question = similar_question.split(" ")
                    similar_question = removeEmptiesAndPunctuation(similar_question)
                    numbers_in_similar_question = findNumbersInWords(similar_question)
                    if len(numbers) == len(numbers_in_similar_question):
                        template_found = True
                        y_pred.append(self.Y_train[index])
                        break
            if not template_found:
                non_sim_index.append(i)
                y_pred.append(-1)
        X_left = [X_test[i] for i in non_sim_index]
        return y_pred, X_left



# In[46]:


json_object_1_f = open('../SVM/data/data.json')
dolphin18K_ds = json.load(json_object_1_f)
json_object_1_f.close()

dolphin18K_ds = [x for x in dolphin18K_ds if len(x['equations']) == 1]

# 14104 questions
TOTAL_QUESTIONS = len(dolphin18K_ds)

# Shuffle and split the dataset
shuffle(dolphin18K_ds)

# Training: 75%, Testing 25%
training_ds = dolphin18K_ds[:int(0.75*TOTAL_QUESTIONS)]
testing_ds = dolphin18K_ds[int(0.75*TOTAL_QUESTIONS):]

def reformat_data_for_CNN(dataset):
    questions = []
    equations = []
    operations = []
    solutions = []
    unknowns = []

    dct = {'*': 'Multiplication', '+': 'Addition', '-': 'Subtraction', '/': 'Division'}

    for qstn_obj in dataset:
        qstn = qstn_obj['question']
        noOfEquations = qstn_obj['noEquations']
        equation = qstn_obj['equations'][0]
        soln = qstn_obj['answers'][0]

        if noOfEquations == 1:
            quants_in_equation = re.findall(r"\d+(?:\.\d+)?", equation)
            if len(quants_in_equation) == 2:
                operator_in_equation = re.findall(r'[-+\/*]', equation)
                if len(operator_in_equation) == 1:
                    unknownInQuestion = re.findall(r"[a-zA-Z]+", equation)
                    if len(unknownInQuestion) > 1:
                        continue

                    qstn = qstn.split()
                    qstn = ' '.join(qstn)
                    questions.append(qstn)
                    equations.append(equation)
                    unknowns.append(unknownInQuestion[0])
                    solutions.append(soln[0])
                    operations.append(dct[operator_in_equation[0]])


    df = pd.DataFrame()
    df['question'] = pd.Series(questions)
    df['noUnknowns'] = pd.Series([1 for _ in range(len(questions))])
    df['unknowns'] = pd.Series(unknowns)
    df['noEquations'] = pd.Series([1 for _ in range(len(questions))])
    df['equations'] = pd.Series(equations)
    df['operation'] = pd.Series(operations)
    df['answers'] = pd.Series(solutions)


    return df


def obtain_train_and_test_for_CNN():
    training_DF_CNN = reformat_data_for_CNN(training_ds)
    testing_DF_CNN = reformat_data_for_CNN(testing_ds)

    return training_DF_CNN, testing_DF_CNN


# In[48]:


def run():
    train, test = obtain_train_and_test_for_CNN()
    X_train, y_train = train['question'].values, train['operation'].values
    X_test, y_test = test['question'].values, test['operation'].values
    print(len(X_train))
    print(len(X_test))

    model = TDIDF(threshold=0.0, num_consider=20)
    model.fit(X_train, y_train)
    y_pred, _ = model.predict(X_test)

    print('Accuracy Score: ', accuracy_score(y_pred, y_test))


# In[49]:


run()
