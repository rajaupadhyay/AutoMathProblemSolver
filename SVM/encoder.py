import json
from encoder_functions import *
from sklearn import svm

fp = open('./data/data.json')
data = json.load(fp)

MIN_EXAMPLE = 2

equationsCount = []
equations = {}
equationsList = []

maxEquationLength = 20
minNumExamples = 20

counter = 0
newData = []

for datapoint in data:
    counter += 1
    words = datapoint['question'].split(' ')
    words = removeEmptiesAndPunctuation(words)
    wordsAndEquations = replaceNumbers(words, datapoint['equations'], datapoint['unknowns'])

    words = wordsAndEquations[0]
    eqTemplates = wordsAndEquations[1]

    #words = addBiTriGrams(words)

    try:
        i = equations[eqTemplates]
        equationsCount[i] += 1
    except:
        i = len(equationsCount)
        equations[eqTemplates] = i
        equationsCount.append(1)
        equationsList.append(eqTemplates)

    newData.append((words, i))


relevantData = []
for datapoint in newData:
    if equationsCount[datapoint[1]] >= MIN_EXAMPLE:
        relevantData.append(datapoint)

vocab = buildVocab(relevantData)
inp = encode(relevantData, vocab)

x = inp[0]
y = inp[1]

clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(x, y)


while(True):
    q = input("Type in a question: ")
    q = q.split(" ")
    q = removeEmptiesAndPunctuation(q)
    numbers = findNumbersInWords(q)
    q = removeEmptiesAndPunctuation(q)
    #q = addBiTriGrams(q)
    inp = encodeTest(q, vocab)
    equationIndex = clf.predict([inp])
    eq = equationsList[equationIndex[0]]
    for i in range(len(numbers)):
        eq = eq.replace("a"+str(i), str(numbers[i]))
    print(eq)
