import json
from encoder_functions import *
from sklearn import svm
from global_params import *
from equivalent_equations import pairEquivalentEquations
import scipy

fp = open('./data/data.json')
data = json.load(fp)

equationsCount = []
equations = {}
equationsList = []

counter = 0
newData = []

for datapoint in data:
    counter += 1
    words = datapoint['question'].split(' ')
    words = removeEmptiesAndPunctuation(words)
    wordsAndEquations = replaceNumbers(words, datapoint['equations'], datapoint['unknowns'])

    words = wordsAndEquations[0]
    eqTemplates = wordsAndEquations[1]

    if len(eqTemplates) <= MAX_EQUATION_LENGTH:
        try:
            i = equations[eqTemplates]
            equationsCount[i] += 1
        except:
            i = len(equationsCount)
            equations[eqTemplates] = i
            equationsCount.append(1)
            equationsList.append(eqTemplates)

        newData.append((words, i))

equations = pairEquivalentEquations(equationsCount, equationsList)

fpout = open('./data/equations.json', 'w')
json.dump(equations, fpout, indent=2)
