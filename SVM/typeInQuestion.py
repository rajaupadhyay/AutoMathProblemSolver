import json
from encoder_functions import *
from sklearn import svm
from global_params import *
from random import shuffle
import SVM
from sympy.solvers import solve
from sympy import Symbol

fp = open('./data/data.json')
data = json.load(fp)


predict = SVM.train(data)

while(True):
    q = input("Type in a question: ")
    print("Equations:")
    equation = predict(q, False)
    print(equation)
    symbols = []
    for char in "xyz":
        if char in equation:
            symbols.append(Symbol(char))
    equation = equation.split(';')
    equation = ['('+e.replace('=', ')-(')+')' for e in equation]
    solution = solve(equation, symbols, minimal=True, quick=True)
    print("Solution:")
    print(solution)
