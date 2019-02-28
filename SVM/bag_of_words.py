import json

fp = open('./data/data.json')
data = json.load(fp)

equationsCount = []
equations = {}

maxEquationLength = 20
num = 0
count = 0
for datapoint in data:
    count+=1
    temp = datapoint['equations'][0]
    temp = temp.replace(datapoint['unknowns'][0], 'x')
    equation = ""
    for char in temp:
        if char == '=':
            equation += '='
        elif char not in "()*/+-x":
            if not equation:
                equation += 'a'
            elif equation[-1] != 'a':
                if equation[-1] not in "+-*/(=":
                    equation += '*a'
                else:
                    equation += 'a'

        else:
            if not equation or char in "+-*/)":
                equation += char
            elif equation[-1] not in "+-*/(=":
                equation += '*' + char
            else:
                equation += char
    if len(equation) > maxEquationLength:
        num+=1
    else:
        try:
            i = equations[equation]
            equationsCount[i] += 1
        except:
            i = len(equations)
            equationsCount.append(1)
            equations[equation] = i
    # print(equation, datapoint['equations'][0])
    # if count>10:
    #     break


print(maxEquationLength, num, len(equationsCount))

equationsCount.sort()

print(equationsCount[-50:])

sum = 0
for a in equationsCount[-50:]:
    sum+=a

print(sum)
