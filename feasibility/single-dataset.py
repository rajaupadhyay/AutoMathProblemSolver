import json

newData = []

fp1 = open('../data/dev_cleaned.json')
fp2 = open('../data/eval_cleaned.json')
fp3 = open('../data/arithmetic.json')
raw1 = json.load(fp1)
raw2 = json.load(fp2)
raw3 = json.load(fp3)

count = 0
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
        if answers[0][0] in ['(A)', '(B)', '(C)', '(D)', '(E)']:
            mcq += 1
        bad = True

    newEntry['answers'] = answers

    if not bad:
        for u in unknowns:
            if len(u) != 1:
                bad = True
                break

    if not bad:
        for option in answers:
            if len(option) != len(unknowns):
                bad = True
                break


    if not bad:
        for option in answers:
            for e in equations:
                temp = '('+e.replace('=', ')-(').replace('^', '**')+')'
                for i in range(len(option)):
                    temp = temp.replace(unknowns[i], str(option[i]))
                try:
                    x = eval(temp)
                    if (not x < 0.1) or (not x > -0.1):
                        bad = True
                except:
                    unsure += 1

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

print(count, mcq, unsure, len(newData), len(raw1+raw2+raw3))

fpout = open('../data/combined.json', 'w')
json.dump(newData, fpout, indent=2)
