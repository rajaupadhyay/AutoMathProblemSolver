import json
from encoder_functions import *
from sklearn import svm
from global_params import *
from random import shuffle

fp = open('./data/data.json')
data = json.load(fp)

fp = open('./data/equations.json')
equations = json.load(fp)

equationsList = ['' for i in range(len(equations))]
for equation in equations:
    equationsList[equations[equation]] = equation

equationsCount = [0 for i in range(len(equations))]

counter = 0
newData = []

for datapoint in data:
    counter += 1
    words = datapoint['question'].split(' ')
    words = removeEmptiesAndPunctuation(words)
    wordsAndEquations = replaceNumbers(words, datapoint['equations'], datapoint['unknowns'])

    # words = wordsAndEquations[0]
    words = addBiTriGrams(words)
    eqTemplates = wordsAndEquations[1]
    if len(eqTemplates) <= MAX_EQUATION_LENGTH:
        i = equations[eqTemplates]
    else:
        i = -1

    newData.append((words, i))



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
            relevantData.append([words, equation])


vocab = buildVocab(relevantData)

inp = encode(relevantData, vocab)

x = inp[0]
y = inp[1]

clf = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo')
clf.fit(x, y)


while(True):
    q = input("Type in a question: ")
    q = q.split(" ")
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
        eq = 'No numerical data given'
    print(eq)
