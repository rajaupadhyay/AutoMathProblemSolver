import random
from sympy.solvers import solve
from sympy import Symbol
from global_params import *
import time


def pairEquivalentEquations(equationsCount, equationsList):
    print("SOLVING EQUATIONS")
    startedAt = time.clock()
    newEquationCount = []
    newEquationsList = []
    newIndeces = [-1 for i in equationsCount]
    testResults = []

    testValues = [12.146652601053638, -12.881259801869144, 10.466388462987638, -17.079292217473792, -7.8910413204445575, -14.476405162737366, -19.133623727067842, -6.323800265702928, -21.551383147476756, 20.97994407224057, 9.661276889741245, -14.500977342116188, -19.27080745399895, -6.484285161326973, 18.159440524055704, 22.63610531349731, 0.31462358163840065, -20.94139504366734]

    for equation in equationsList:

        modified = "(" + equation.replace("=", ")-(") + ")"

        i = 0
        aVector = []
        while 'a' in modified:
            name = 'a'+str(i)
            if name in modified:
                newName = 'b'+str(i)
                modified = modified.replace(name, newName)
                aVector.append(1)
            aVector.append(0)
            i += 1


        while len(testValues) < len(aVector):
            testValues.append(random.random()*50-25)

        for j in range(i):
            modified = modified.replace("b"+str(j), str(testValues[j]))

        x = Symbol('x')
        solution = solve(modified, x)
        try:
            testResults.append([solution[0], aVector])
        except:
            testResults.append([None, aVector])

    print("took: ", time.clock()-startedAt,"s")
    print("MATCHING EQUATIONS")
    startedAt = time.clock()

    newEquations = {}
    for i in range(len(equationsList)):
        if newIndeces[i] == -1:
            ind = len(newEquationCount)
            newEquations[equationsList[i]] = ind
            newEquationsList.append(equationsList[i])
            newEquationCount.append(equationsCount[i])
            if testResults[0] != None and 1 in testResults[i][1]:
                for j in range(i+1, len(equationsList)):
                    if newIndeces[j] == -1:
                        if testResults[i][0] == testResults[j][0] and testResults[i][1] == testResults[j][1]:
                            newIndeces[j] = ind
                            newEquationCount[-1] += equationsCount[j]
                            newEquations[equationsList[i]] = ind

    print("took: ", time.clock()-startedAt,"s")
    print("Reduced from ", len(equationsList)," equations to ", len(newEquationsList), " equations")
    return newEquations
