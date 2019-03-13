import json
from encoder_functions import *
from sklearn import svm
from global_params import *
from random import shuffle

fp = open('./data/data.json')
data = json.load(fp)

fp = open('./data/equations.json')
equations = json.load(fp)

counter = 0
newData = []

maxIndex = 0

for i in equations.values():
    if i > maxIndex:
        maxIndex = i

equationsList = ['' for i in range(maxIndex+1)]
for equation in equations:
    equationsList[equations[equation]] = equation

equationsCount = [0 for i in range(maxIndex+1)]

for datapoint in data:
    if datapoint['noEquations'] <= MAX_NO_EQUATIONS:
        counter += 1
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


count = 0
for dp in testData:
    q = dp[0]
    inp = encodeTest(q, vocab)
    numbers = dp[3]
    equationIndex = clf.predict([inp])
    eq = equationsList[equationIndex[0]]
    answers = dp[2]

    for i in range(len(numbers)):
        eq = eq.replace("a"+str(i), str(numbers[i]))
    if numbers:
        i = len(numbers)
        while 'a' in eq:
            eq = eq.replace("a"+str(i), str(numbers[-1]))
            i+=1
    else:
        eq = ''

    correct = False

    if eq:
        for answer in answers:
            possibleUnknowns = "xyz"
            permutations = [[[0]],[[0,1],[1,0]],[[0,1,2],[1,0,2],[0,2,1],[1,2,0],[2,1,0],[2,0,1]]]
            for permutation in permutations[len(answer)-1]:
                tempEq = eq
                for i in range(len(answer)):
                    tempEq = tempEq.replace(possibleUnknowns[permutation[i]], str(answer[i]))
                tempEq = tempEq.split(';')
                correctTemp = True
                for teq in tempEq:
                    try:
                        sol = eval('('+ teq.replace('=', ')-(') +')')
                        if sol > ZERO or sol < -ZERO:
                            correctTemp = False
                    except:
                        correctTemp = False

                correct = correct or correctTemp
                if correct:
                    break
            if correct:
                break

    if correct:
        count += 1


print(count/len(testData))
