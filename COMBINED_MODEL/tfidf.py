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
# sys.path.insert(0, '../SVM')
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


class TFIDF:

    def __init__(self, threshold, num_consider, max_df):
        self.threshold = threshold
        self.num_consider = num_consider
        self.max_df = max_df


    def get_tfidf_matrix(self, corpus):
        self.tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), min_df=0, max_df=self.max_df)
        self.tfidf_matrix = self.tf.fit_transform(corpus)


    # def replace_nouns(self, corpus):
    #     for j in range(len(corpus)):
    #         pos = pos_tag(word_tokenize(corpus[j]))
    #         for i in range(len(pos)):
    #             if pos[i][0].lower() in MATH_TERMS:
    #                 continue
    #             if pos[i][1] == 'NN' or pos[i][1] == 'NNS':
    #                 corpus[j] = str.replace(corpus[j], pos[i][0], 'CN')
    #             elif pos[i][1] == 'NNP' or pos[i][1] == 'NNPS':
    #                 corpus[j] = str.replace(corpus[j], pos[i][0], 'PN')
    #     return corpus
    def replace_nouns(self, corpus, window_size = 3):
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

                if pos[i][0].lower() in MATH_TERMS:
                    continue

                if pos[i][1] == 'NN' or pos[i][1] == 'NNS':
                    corpus[j] = str.replace(corpus[j], pos[i][0], 'CN')
        return corpus


    # Find documents similar to a document that is not in the tfidf_matrix
    def find_similar(self, new_doc):
        cosine_similarities = linear_kernel(new_doc[0], self.tfidf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
        return [(index, cosine_similarities[index]) for index in related_docs_indices][0:self.num_consider]


    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        X_train = self.replace_nouns(X_train)
        print(X_train)

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
                    # if len(numbers) == len(numbers_in_similar_question):
                    template_found = True
                    y_pred.append(self.Y_train[index])
                    break
            if not template_found:
                non_sim_index.append(i)
                # y_pred.append(-1)
        X_left = [X_test[i] for i in non_sim_index]

        fp = open('SVM/data/equations.json')
        equation_dict = json.load(fp)

        y_pred = [key for x in y_pred for key, value in equation_dict.items() if value == x]

        if len(y_pred) > 0:
            for i in range(len(numbers)):
                y_pred[0] = y_pred[0].replace("a" + str(i), str(numbers[i]))

        return y_pred, X_left
