import json
from encoder_functions import *
from sklearn import svm
from global_params import *
from random import shuffle


fp = open('./data/equations.json')
equations = json.load(fp)

fp = open('./data/equationsList.json')
equationsList = json.load(fp)

fp = open('./data/lowPerformingIndices.json')
lowPerformingIndices = json.load(fp)


def train(data):
    newData = []

    for datapoint in data:
        equationsCount = [0 for i in equationsList]
        if datapoint['noEquations'] <= MAX_NO_EQUATIONS:
            words = datapoint['question'].split(' ')
            words = removeEmptiesAndPunctuation(words)
            wordsAndEquations = replaceNumbers(words, datapoint['equations'], datapoint['unknowns'])

            words = wordsAndEquations[0]
            numbers = wordsAndEquations[2]
            words = addBiTriGrams(words)
            eqTemplates = wordsAndEquations[1]
            i = -1
            if len(eqTemplates) <= MAX_EQUATION_LENGTH:
                try:
                    i = equations[eqTemplates]
                except:
                    maxIndex += 1
                    i = maxIndex
                    equations[eqTemplates] = i
                    equationsCount.append(0)
                    equationsList.append(eqTemplates)

            newData.append((words, i, datapoint['answers'], numbers))


    shuffle(newData)

    twentyFive = int(0.25 * len(newData))

    testData = newData[:twentyFive]
    newData = newData[twentyFive:]

    for datapoint in newData:
        equation = datapoint[1]
        if equation != -1:
            equationsCount[equation] += 1

    relevantData = []


    for datapoint in newData:
        words = datapoint[0]
        equation = datapoint[1]
        if equation != -1:
            if equationsCount[equation] >= MIN_EXAMPLE:
                relevantData.append([words, equation, datapoint[2], datapoint[3]])


    vocab = buildVocab(relevantData)

    inp = encode(relevantData, vocab)

    x = inp[0]
    y = inp[1]

    clf = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo')
    clf.fit(x, y)
    return lambda x: fitToModel(clf, vocab, x)



def fitToModel(clf, vocab, question):
    q = question.split(" ")
    q = removeEmptiesAndPunctuation(q)
    numbers = findNumbersInWords(q)
    q = removeEmptiesAndPunctuation(q)
    q = addBiTriGrams(q)
    inp = encodeTest(q, vocab)
    equationIndex = clf.predict([inp])
    if lowPerformingIndices[equationIndex[0]]:
        return False
    eq = equationsList[equationIndex[0]]

    for i in range(len(numbers)):
        eq = eq.replace("a"+str(i), str(numbers[i]))
    if numbers:
        i = len(numbers)
        while 'a' in eq:
            eq = eq.replace("a"+str(i), str(numbers[-1]))
            i+=1
    else:
        eq = False

    return eq
