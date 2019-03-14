import json
from encoder_functions import *
from sklearn import svm
from global_params import *
from random import shuffle
from checkSolution import *
import SVM

fp = open('./data/data.json')
data = json.load(fp)

fp = open('./data/equations.json')
equations = json.load(fp)

for iteration in range(10):
    twentyFive = int(0.25*len(data))

    shuffle(data)
    test = data[:twentyFive]
    train = data[twentyFive:]

    predict = SVM.train(train)

    right = 0

    for datapoint in test:
        predicted = predict(datapoint['question'], False)
        if checkSolution(predicted, datapoint['answers']):
            right += 1

    print(right/twentyFive)
