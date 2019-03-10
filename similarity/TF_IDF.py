#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import json
import string
from nltk import pos_tag
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import sys
sys.path.insert(0, '../SVM')
from encoder_functions import *


# In[2]:


def remove_specials_and_spaces(s):
    # Removing punctuations
    stripped = re.sub('[?!.,;:_]', '', s)
    
    # Removing excess white space
    stripped = re.sub('\s+', ' ', stripped)
    stripped = stripped.strip()

    return stripped


# In[3]:


def get_corpus(filepath = '../SVM/data/data.json'):
    corpus = []
    with open(filepath) as file:
        data = json.load(file)
    for problem in data:
        corpus.append(problem['question'])
    return corpus


# ### POS Tagging
# NN	  noun, singular 'desk'  
# NNS	  noun plural	'desks'  
# NNP	  proper noun, singular	'Harrison'  
# NNPS  proper noun, plural	'Americans'  

# In[4]:


# http://lbcrs.org/common/pages/DisplayFile.aspx%3FitemId%3D3446744
# https://www.purplemath.com/modules/translat.htm
math_terms = [
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


# In[5]:

def replace_nouns(corpus):
    for j in range(len(corpus)):
        pos = pos_tag(word_tokenize(corpus[j]))
        for i in range(len(pos)):
            if pos[i][0].lower() in math_terms:
                continue
            if pos[i][1] == 'NN' or pos[i][1] == 'NNS':
                corpus[j] = str.replace(corpus[j], pos[i][0], 'commonnoun')
            elif pos[i][1] == 'NNP' or pos[i][1] == 'NNPS':
                corpus[j] = str.replace(corpus[j], pos[i][0], 'propernoun')
    return corpus

#def replace_nouns(corpus, window_size = 2):
#    for j in range(len(corpus)):
#        count = window_size + 1
#        pos = pos_tag(word_tokenize(corpus[j]))
#        for i in range(len(pos)):
#            
#            try:
#                float(pos[i][0])
#                count = 0
#                continue
#            except ValueError:
#                count += 1
#                
#            if (pos[i][1] == 'NNP' or pos[i][1] == 'NNPS') and i > 0:
#                corpus[j] = str.replace(corpus[j], pos[i][0], 'propernoun')
#                continue
#            
#            if count > window_size:
#                continue;
#                
#            if pos[i][0].lower() in math_terms:
#                continue
#            
#            if pos[i][1] == 'NN' or pos[i][1] == 'NNS':
#                corpus[j] = str.replace(corpus[j], pos[i][0], 'commonnoun')
#    return corpus


# In[6]:


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


# In[7]:


def redundant_equation_remover(corpus, equations_list, min_app = 2):
    equations_dict = dict()
    for equation in equations_list:
        if equation in equations_dict:
            equations_dict[equation] += 1
        else:
            equations_dict[equation] = 1
#     print(equations_list)
    for i in range(len(equations_list) - 1, -1, -1):
        if equations_dict[equations_list[i]] < min_app:
#             print(i)
            del equations_list[i]
            del corpus[i]
    return corpus, equations_list


# In[8]:


def get_tfidf_matrix(corpus):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(corpus)
    return tf, tfidf_matrix


# In[9]:


# Find documents similar to another document in the tfidf_matrix at given index
def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

# Find documents similar to a document that is not in the tfidf_matrix
def find_similar_new(tfidf_matrix, new_doc, top_n = 2):
    cosine_similarities = linear_kernel(new_doc[0], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]


# In[10]:


def print_similar_questions(corpus, tf, tfidf_matrix, min_score):
    count = 0
    for i in range(len(corpus)):
        for index, score in find_similar(tfidf_matrix, i, top_n=1):
            if score > 0.5:
                count += 1
                print('Question', i)
                print(corpus[i])
                print('Similarity Score: ', score)
                print('Similar Question: ', corpus[index], '\n')
    print('Number of problems with score above', min_score, ': ', count)
    print('Total number of problems in corpus: ', len(corpus))
    print('Fraction of entire corpus: ', count/len(corpus))


# In[11]:


def get_input_and_similar_questions(corpus, equations_list, tf, tfidf_matrix, min_score):
    while True:
        question = input("Type in a question: ")
        temp = question.split(" ")
        temp = removeEmptiesAndPunctuation(temp)
        if question.lower() == 'exit':
            print('Exitting...\n')
            break
        
        numbers = findNumbersInWords(temp)
        question = replace_nouns([question])[0]
        print(question)
        new_doc = tf.transform([question])
        
        template_found = False
        
        for index, score in find_similar_new(tfidf_matrix, new_doc, top_n=15):
            if score > min_score:
                similar_question = corpus[index]
                similar_question = similar_question.split(" ")
                similar_question = removeEmptiesAndPunctuation(similar_question)
                numbers_in_similar_question = findNumbersInWords(similar_question)
                if len(numbers) == len(numbers_in_similar_question):
                    print(len(numbers_in_similar_question))
                    print(numbers_in_similar_question)
                    print(numbers)
                    template_found = True
                    equation = equations_list[index]
                    for i in range(len(numbers)):
                        equation = equation.replace("a" + str(i), str(numbers[i]))
                    print(equation)
                    print('-------------------------------------')
                    print('Similarity Score: ', score)
                    print('Similar Question: ', corpus[index], '\n\n')
                    break
        if not template_found:
            print('No similar questions found!\n')


# In[12]:


def user_run():
    min_score = 0.0
    filepath = '../SVM/data/data.json'
    
    corpus = get_corpus(filepath)
    corpus = replace_nouns(corpus)
    equations_list = prep_equation_list(filepath)
#     corpus, equations_list = redundant_equation_remover(corpus, equations_list, min_app = 2)
    tf, tfidf_matrix = get_tfidf_matrix(corpus)
    get_input_and_similar_questions(corpus, equations_list, tf, tfidf_matrix, min_score)
#     print_similar_questions(corpus, tf, tfidf_matrix, min_score)


# In[13]:


user_run()

