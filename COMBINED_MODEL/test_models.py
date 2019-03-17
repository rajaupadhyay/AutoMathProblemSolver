import pandas as pd
from train_models import train_all_models
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
from SVM import fitToModel
from checkSolution import checkSolution
import re

## CNN HELPER FUNCTIONS
def retrieve_one_hot_embeddings(questions, word_to_idx):
    embeddings = []
    for qstn in questions:
        embeddings.append([word_to_idx[word] for word in qstn.split() if word in word_to_idx])
    return embeddings


def returnEquationsAndSolutions(testDS, predictedOperators):
    zippedQuestionsAndOps = list(zip(testDS, predictedOperators))

    equationsOutput = []

    for ques, op in zippedQuestionsAndOps:
        quantities = re.findall(r"\d+(?:\.\d+)?", ques)

        quantities = list(map(float, quantities))

        if len(quantities) != 2:
            return [-1]

        # op_val = {'Addition': '+', 'Multiplication': '*', 'Division': '/', 'Subtraction': '-'}

        eqtn_ = None
        if op == 'Addition':
            eqtn_ = 'x={}{}{}'.format(quantities[0], '+', quantities[1])
        elif op == 'Multiplication':
            eqtn_ = 'x={}{}{}'.format(quantities[0], '*', quantities[1])
        elif op == 'Division':
            lhs = max(quantities)
            rhs = min(quantities)
            eqtn_ = 'x={}{}{}'.format(lhs, '/', rhs)
        else:
            lhs = max(quantities)
            rhs = min(quantities)
            eqtn_ = 'x={}{}{}'.format(lhs, '-', rhs)

        equationsOutput.append(eqtn_)

    return equationsOutput


def loadAndTestCNNModel(testDS, vocab_dict, model):
    vocabulary_size = len(vocab_dict)+1

    testDS_embedding = retrieve_one_hot_embeddings(testDS, vocab_dict)

    labels = ['Addition', 'Division', 'Multiplication', 'Subtraction']

    # Create dict for labels to index
    label_to_index = {o:i for i,o in enumerate(labels)}
    index_to_label = {i:o for i,o in enumerate(labels)}


    max_length = 100
    testDS_embedding_padded = pad_sequences(testDS_embedding, maxlen=max_length, padding='post')



    y_predict = model.predict(testDS_embedding_padded)
    # Arg max to get the predicted operator
    y_predict = [index_to_label[np.argmax(i)] for i in y_predict]

    dictionaryOfOperations = {'Addition': '+', 'Subtraction': '-', 'Division': '/', 'Multiplication': '*'}

    return dictionaryOfOperations[y_predict[0]]


avgAccuracy = 0
for itx in range(1):

    svmFitFunction, tfidf_mdl, testGlobal = train_all_models()
    totalQs = len(testGlobal)
    print('Testing on {} samples'.format(totalQs))



    TOTALCORRECT = 0
    TFIDF_attempts = 0
    TFIDF_CORRECT = 0
    chcker = 0
    for questionObject in testGlobal:
        if chcker%500 == 0:
            print('Current itx {}/{}'.format(chcker, totalQs))
        # if chcker % 100 == 0:
            # print('Test sample {} - correct {}'.format(chcker, TOTALCORRECT))


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
            res = svmFitFunction(question, False)
            correctionCheck = checkSolution(res, answer)


        if correctionCheck:
            TOTALCORRECT += 1

        # equation = None
        #
        # if isSingle:
        #     predictedOperator = loadAndTestCNNModel([question], vocab_dict, CNN_model)
        #     if predictedOperator in res:
        #         equation = res
        #     else:
        #         try:
        #             # equation = svmFitFunctionWithAdvice(question, predictedOperator)
        #             for op in "+-/*":
        #                 res = res.replace(op, predictedOperator)
        #             equation = res
        #         except:
        #             equation = res
        #
        #     questionsCNN.append(question)
        #     solutionsForQsCNN.append(answer[0])
        #     predictedOperatorCNN.append(predictedOperator)
        #     equation = returnEquationsAndSolutions([question], [predictedOperator])[0]
        # else:
        #     equation = res
        #
        # if equation == -1:
        #     equation = svmFitFunction(question, False)


    print('Iteration ', itx+1)
    # print('Total Correct', TOTALCORRECT)
    # print('Testing on {} samples'.format(len(testGlobal)))

    avgAccuracy += TOTALCORRECT/len(testGlobal)
    print('Overall accuracy:', TOTALCORRECT/len(testGlobal))
    print('TFIDF Attempted {} questions'.format(TFIDF_attempts))
    print('TFIDF accuracy ', TFIDF_CORRECT/TFIDF_attempts)

print('Average Accuracy over 10 runs', avgAccuracy/5)
