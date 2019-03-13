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

fpout = open('./data/equationsList.json', 'w')
json.dump(equationsList, fpout, indent=2)


lowPerformingIndices = [False for i in range(maxIndex+1)]
for i in range(12):
    for j in range(12):
        for op in "-/*+":
            try:
                ind = equations['x=a'+str(i)+op+'a'+str(j)]
                lowPerformingIndices[ind] = True
            except:
                pass

fpout = open('./data/lowPerformingIndices.json', 'w')
json.dump(lowPerformingIndices, fpout, indent=2)
