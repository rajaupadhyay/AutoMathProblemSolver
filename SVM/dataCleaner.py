import json
from math import *

newData = []

fp1 = open('../data/dev_cleaned.json')
fp2 = open('../data/eval_cleaned.json')
fp3 = open('../data/arithmetic.json')
raw1 = json.load(fp1)
raw2 = json.load(fp2)
raw3 = json.load(fp3)

count = 0
nonNum = 0
wrongSolution = 0
inconsistentUnknowns = 0
numbersOfEquationsNoUnk = 0
tooManyEq = 0
moreThanThreeEq = 0
emptyQuestion = 0
emptyUnknowns = 0
emptyEquations = 0
mcq = 0
unsure = 0

for entry in raw1+raw2:
    newEntry = {}
    newEntry['question'] = entry['text']
    unknownsAndEquations = entry['equations'].split('\r\n')

    unknowns = unknownsAndEquations[0].replace('unkn:', '').replace(' ', '').split(',')
    noUnknowns = len(unknowns)
    newEntry['noUnknowns'] = len(unknowns)
    newEntry['unknowns'] = unknowns

    equations = unknownsAndEquations[1:]
    equations = [e.replace('equ:', '').replace(' ', '') for e in equations]
    newEntry['noEquations'] = len(equations)
    newEntry['equations'] = equations

    answers = entry['ans']
    answers = answers.replace('{','').replace('}','').replace(' ', '').split('|')[-1].split('or')
    answers = [option.split(';') for option in answers]

    bad = False

    try:
        answers = [[float(eval(o)) for o in option] for option in answers]
    except:
        nonNum += 1
        if answers[0][0] in ['(A)', '(B)', '(C)', '(D)', '(E)']:
            mcq += 1
        bad = True

    newEntry['answers'] = answers

    if not bad:
        while True:
            try:
                unkowns.remove('')
            except:
                break
        if len(unknowns) == 0:
            emptyUnknowns +=1
            bad = True
            break

    if not bad:
        for option in answers:
            if len(option) != len(unknowns):
                inconsistentUnknowns += 1
                bad = True
                break


    if not bad:
        for option in answers:
            for e in equations:
                temp = '('+e.replace('=', ')-(')+')'
                charset1a = unknowns + [')']
                charset2b = unknowns + ['(']
                charset2a = charset1a + [str(i) for i in range(10)]
                charset1b = charset2b + [str(i) for i in range(10)]
                for a in charset1a:
                    for b in charset1b:
                        temp = temp.replace(a+b, a+'*'+b)
                for a in charset2a:
                    for b in charset2b:
                        temp = temp.replace(a+b, a+'*'+b)
                for i in range(len(option)):
                    temp = temp.replace(unknowns[i], str(option[i])).replace('^', '^^')
                try:
                    x = eval(temp)
                    if (not x < 0.1) or (not x > -0.1):
                        wrongSolution += 1
                        #ADD BETTER CHECKER IF NEED BE
                        bad = True
                except:
                    unsure += 1
                    bad = True
    if not bad:
        if newEntry['question'] == "":
            emptyQuestion += 1
            bad = True
    if not bad:
        if newEntry['noEquations'] == 0:
            emptyEquations += 1
            bad = True

    if not bad and newEntry['noUnknowns'] != newEntry['noEquations']:
        bad = True
        numbersOfEquationsNoUnk += 1

    if not bad and newEntry['noEquations'] > 3:
        tooManyEq += 1
        bad = True


    if bad:
        count += 1
    else:
        newData.append(newEntry)


for entry in raw3:
    bad = False
    equations = entry['lEquations']
    equations = [e.replace('X','x').replace(' ', '') for e in equations]
    answers = [[float(entry['lSolutions'][0])]]
    noEquations = len(entry['lEquations'])
    try:
        noUnknowns = len(entry['lQueryVars'])
        unknowns = entry['lQueryVars']
    except:
        noUnknowns = 1
        unknowns = ['x']
    newEntry = {
        'question' : entry['sQuestion'],
        'noUnknowns': noUnknowns,
        'unknowns' : unknowns,
        'noEquations': noEquations,
        'equations' : equations,
        'answers' : answers
    }

    if noUnknowns != noEquations:
        bad = True
        numbersOfEquationsNoUnk += 1

    if not bad and noEquations > 3:
        tooManyEq +=1
        bad = True

    if bad:
        count += 1
    else:
        newData.append(newEntry)
print("Original Size: %d (%d , %d, %d)" % (len(raw1+raw2+raw3), len(raw1), len(raw2), len(raw3)))
print("Size of New Dataset: ", len(newData))
print("Bad Examples: ", count)
print("  Non Numerical: ", nonNum)
print("    out of which mcq: ", mcq)
print("  Inconsistent Unkowns: ", inconsistentUnknowns)
print("  Number Unkowns != Number of Equations: ", numbersOfEquationsNoUnk)
print("  >3 Equations: ", tooManyEq)
print("  Empty Question: ", emptyQuestion)
print("  No Equations: ", emptyEquations)
print("  Bad Solution: ", wrongSolution)
print("Unsure of Correctness of Solution: ", unsure)

fpout = open('./data/data.json', 'w')
json.dump(newData, fpout, indent=2)
