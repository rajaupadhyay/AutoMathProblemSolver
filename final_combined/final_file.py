import pandas as pd
import json
import re
from random import shuffle
import numpy as np
from helper_functions import *
# from svm import train
from tfidf import *
import pickle


# TFIDF
import string
import random
from nltk import pos_tag
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


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

        fp = open('final_combined/data/equations.json')
        equation_dict = json.load(fp)

        y_pred = [key for x in y_pred for key, value in equation_dict.items() if value == x]

        if len(y_pred) > 0:
            for i in range(len(numbers)):
                y_pred[0] = y_pred[0].replace("a" + str(i), str(numbers[i]))

        return y_pred, X_left






json_object_1_f = open('final_combined/data/data.json')
dolphin18K_ds = json.load(json_object_1_f)
json_object_1_f.close()


fp = open('final_combined/data/equations.json')
equations = json.load(fp)

fp = open('final_combined/data/equationsList.json')
equationsList = json.load(fp)


# dolphin18K_ds = [x for x in dolphin18K_ds if x['noEquations'] == 1]

# 14104 questions
TOTAL_QUESTIONS = len(dolphin18K_ds)

# Shuffle and split the dataset
shuffle(dolphin18K_ds)

# Training: 75%, Testing 25%
training_ds = dolphin18K_ds[:int(0.75*TOTAL_QUESTIONS)]
testing_ds = dolphin18K_ds[int(0.75*TOTAL_QUESTIONS):]



def fitToModel(clf, vocab, question, requireSingleOp):
    q = question.split(" ")
    q = removeEmptiesAndPunctuation(q)
    numbers = findNumbersInWords(q)
    q = removeEmptiesAndPunctuation(q)
    q = addBiTriGrams(q)
    inp = encodeTest(q, vocab)
    equationIndex = clf.predict([inp])
    eq = equationsList[equationIndex[0]]

    for i in range(len(numbers)):
        eq = eq.replace("a"+str(i), str(numbers[i]))
    if numbers:
        i = len(numbers)
        while 'a' in eq:
            eq = eq.replace("a"+str(i), str(numbers[-1]))
            i+=1
    else:
        eq = ''

    return eq



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
    fp = open('final_combined/data/equations.json')
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

    return corpus, equation_values



def obtain_train_and_test():
    train_synthetic = pd.read_csv('final_combined/data/train_synthetic.csv', sep=',')
    tfidfX_train, tfidfy_train = createTfidfTrainAndTest(train_synthetic)

    return tfidfX_train, tfidfy_train, training_ds, testing_ds



tfidfX_train, tfidfy_train, training_ds, testing_ds = obtain_train_and_test()

# svmFitFunction, _ = train(training_ds)

svm_model_f = open('final_combined/svm_clf.pickle', 'rb')
svm_model = pickle.load(svm_model_f)
svm_model_f.close()

svm_vocab_f = open('final_combined/svm_vocab.pickle', 'rb')
svm_vocab = pickle.load(svm_vocab_f)
svm_vocab_f.close()


tfidf_mdl = TFIDF(0.6, 5, 1.0)
tfidf_mdl.fit(tfidfX_train, tfidfy_train)


svmFitFunction = fitToModel

testGlobal = testing_ds

avgAccuracy = 0
for itx in range(5):

    totalQs = len(testGlobal)
    print('Testing on {} samples'.format(totalQs))

    TOTALCORRECT = 0
    TFIDF_attempts = 0
    TFIDF_CORRECT = 0
    chcker = 0
    correctionCheck = False
    for questionObject in testGlobal:
        if chcker%500 == 0:
            print('Current itx {}/{}'.format(chcker, totalQs))

        chcker += 1
        question = questionObject['question']
        answer = questionObject['answers']

        equationTFIDF, _ = tfidf_mdl.predict([question])

        correctVal = 0

        if len(equationTFIDF)>0:
            equationTFIDF = equationTFIDF[0]
            correctionCheck = checkSolution(equationTFIDF, answer)
            if correctionCheck:
                TFIDF_CORRECT += 1
            TFIDF_attempts += 1
        else:
            res = svmFitFunction(svm_model, svm_vocab, question, False)
            correctionCheck = checkSolution(res, answer)

        if correctionCheck:
            TOTALCORRECT += 1


    print('Iteration ', itx+1)


    avgAccuracy += TOTALCORRECT/len(testGlobal)
    print('Overall accuracy:', TOTALCORRECT/len(testGlobal))
    print('TFIDF Attempted {} questions'.format(TFIDF_attempts))
    print('TFIDF accuracy ', TFIDF_CORRECT/TFIDF_attempts)

print('Average Accuracy over 5 runs', avgAccuracy/5)
