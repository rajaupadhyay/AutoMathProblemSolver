import pandas as pd
import json
import re
from random import shuffle
import numpy as np
from helper_functions import *
# from svm import train
from tfidf import *
import pickle












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
