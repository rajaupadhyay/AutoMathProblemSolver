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
emptyQuestion = 0
emptyUnknowns = 0
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
        for u in unknowns:
            if len(u) != 1:
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
                temp = '('+e.replace('=', ')-(').replace('^', '**')+')'
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
                    temp = temp.replace(unknowns[i], str(option[i]))
                try:
                    x = eval(temp)
                    if (not x < 0.1) or (not x > -0.1):
                        wrongSolution += 1
                        bad = True
                except:
                    unsure += 1
    if newEntry['question'] == "":
        emptyQuestion += 1
        bad = True
    if bad:
        count += 1
    else:
        newData.append(newEntry)

for entry in raw3:
    equations = entry['lEquations']
    equations = [e.replace('X','x').replace(' ', '') for e in equations]
    answers = [[float(entry['lSolutions'][0])]]

    newEntry = {
        'question' : entry['sQuestion'],
        'noUnknowns': 1,
        'unknowns' : ['x'],
        'noEquations': 1,
        'equations' : entry['lEquations'],
        'answers' : answers
    }

    newData.append(newEntry)
print("Original Size: ", len(raw1+raw2+raw3))
print("Size of New Dataset: ", len(newData))
print("Bad Examples: ", count)
print("  Non Numerical: ", nonNum)
print("    out of which mcq: ", mcq)
print("  Empty Unkowns: ", emptyUnknowns)
print("  Empty Question: ", emptyQuestion)
print("  Bad Solution: ", wrongSolution)
print("Unsure of Correctness of Solution: ", unsure)

fpout = open('../data/combined.json', 'w')
json.dump(newData, fpout, indent=2)
