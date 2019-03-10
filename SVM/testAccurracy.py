import json
from encoder_functions import *
from sklearn import svm
from global_params import *
from random import shuffle

fp = open('./data/data.json')
data = json.load(fp)

fp = open('./data/equations.json')
equations = json.load(fp)

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
            relevantData.append([words, equation])


vocab = buildVocab(relevantData)

inp = encode(relevantData, vocab)

x = inp[0]
y = inp[1]

clf = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo')
clf.fit(x, y)


count = 0
for dp in testData:
    q = dp[0]
    numbers = findNumbersInWords(q)
    q = removeEmptiesAndPunctuation(q)
    q = removeEmptiesAndPunctuation(q)
    numbers = findNumbersInWords(q)
    q = removeEmptiesAndPunctuation(q)
    inp = encodeTest(q, vocab)
    equationIndex = clf.predict([inp])

    if equationIndex == dp[1]:
        count += 1

print(count/len(testData))
