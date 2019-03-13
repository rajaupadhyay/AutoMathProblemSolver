import json
from encoder_functions import *
from sklearn import svm
from global_params import *
from random import shuffle
import SVM

fp = open('./data/data.json')
data = json.load(fp)

fp = open('./data/equations.json')
equations = json.load(fp)


predict = SVM.train(data)

print(predict("John has five apples and three pears. How many pears does he have?", False))
print(predict("John has five apples and three pears. How many pears does he have?", True))
print(predict("The product of two numbers is 15 and their sum is 14. What are the two numbers?", True))
print(predict("Not a question.", False))
print(predict("Not a question.", True))
