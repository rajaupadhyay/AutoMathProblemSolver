import json
from helper_functions import *
from sklearn import svm
from random import shuffle


fp = open('data/equations.json')
equations = json.load(fp)

fp = open('data/equationsList.json')
equationsList = json.load(fp)

# fp = open('COMBINED_MODEL/svm_data/lowPerformingIndices.json')
# lowPerformingIndices = json.load(fp)


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
                    # print(datapoint)
                    i = len(equationsList)
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
    f1 = lambda x,y: fitToModel(clf, vocab, x,y)
    f2 = lambda x,y: guidedFitToModel(clf, vocab, x, y)
    return (f1, f2)



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

    # if requireSingleOp:
    #     singleOp = False
    #     try:
    #         singleOp = lowPerformingIndices[equationIndex[0]]
    #     except:
    #         pass
    #     return (eq, singleOp)

    return eq

def guidedFitToModel(clf, vocab, question, operation):
    q = question.split(" ")
    q = removeEmptiesAndPunctuation(q)
    numbers = findNumbersInWords(q)
    q = removeEmptiesAndPunctuation(q)
    q = addBiTriGrams(q)
    inp = encodeTest(q, vocab)
    scores = clf.decision_function([inp])

    possibleEquations = []
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            equationTemplate = "x=a"+str(i)+operation+"a"+str(j)
            # print(equationTemplate)
            try:
                equationIndex = equations[equationTemplate]
                possibleEquations.append(clf.classes_.index(equationIndex))
            except:
                equationTemplate = "a"+str(i)+operation+"a"+str(j)+'=x'
                try:
                    equationIndex = equations[equationTemplate]
                    possibleEquations.append(clf.classes_.index(equationIndex))
                except:
                    pass


    possibleEquations.sort()

    vote = [0 for _ in possibleEquations]

    n = len(clf.classes_)
    for i in range(len(possibleEquations)):
        for j in range(i+1,len(possibleEquations)):
            I = possibleEquations[i]
            J = possibleEquations[j]
            if scores[I*n - (I+1)*I/2] < 0:
                vote[j] += 1
            else:
                vote[i] += 1

    maxIndex = vote.index(max(vote))

    eq = equationsList[maxIndex]

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
