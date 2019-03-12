import random
from sympy.solvers import solve
from sympy import Symbol
from global_params import *
import sys
import time
from multiprocessing import Process, Pipe

def solveMe(modified, symbols, con):
    solution = None
    try:
        solution = solve(modified, symbols, minimal=True, quick=True, dict=False)
    except:
        solution = None
    try:
        solution = list(solution[0])

    except:
        try:
            solution = list(solution.values())
        except:
            solution = solution
    finally:
        try:
            solution = [float(s) for s in solution]
        except:
            solution = None
    con.send(solution)
    con.close()



def pairEquivalentEquations(equationsCount, equationsList):
        print("SOLVING EQUATIONS")
        startedAt = time.clock()
        newEquationCount = []
        newEquationsList = []
        newIndeces = [-1 for i in equationsCount]
        testResults = []

        testValues1 = []
        testValues2 = []


        prog = 0
        for equation in equationsList:
            prog+=1
            print("%.2f%%" % (100*prog/len(equationsList)))
            sys.stdout.write("\033[F")


            possibleUnknowns = "xyz"

            modified = equation

            i = 0
            aVector = []

            while 'a' in modified:
                name = 'a'+str(i)
                if name in modified:
                    newName = 'b'+str(i)
                    modified = modified.replace(name, newName)
                    aVector.append(1)
                else:
                    aVector.append(0)
                i += 1


            while len(testValues1) < len(aVector):
                testValues1.append(random.randint(-50,50))
                testValues2.append(random.randint(-50,50))

            modified1 = modified
            modified2 = modified
            for j in range(i):
                modified1 = modified1.replace("b"+str(j), str(testValues1[j]))
                modified2 = modified2.replace("b"+str(j), str(testValues2[j]))

            modified1 = modified1.split(';')
            modified1 = ['('+m.replace('=', ')-(')+')' for m in modified1]
            symbols = [Symbol(possibleUnknowns[i]) for i in range(len(modified1))]
            parent_conn, child_conn = Pipe()
            solution1 = None
            p = Process(target=solveMe, args=(modified1, symbols, child_conn))
            p.start()
            p.join(timeout=5)
            if not p.is_alive():
                solution1 = parent_conn.recv()
            p.terminate()

            modified2 = modified2.split(';')
            modified2 = ['('+m.replace('=', ')-(')+')' for m in modified2]
            symbols = [Symbol(possibleUnknowns[i]) for i in range(len(modified2))]
            parent_conn, child_conn = Pipe()
            solution2 = None
            p = Process(target=solveMe, args=(modified2, symbols, child_conn))
            p.start()
            p.join(timeout=5)
            if not p.is_alive():
                solution2 = parent_conn.recv()
            p.terminate()

            toAppend = []
            if solution1:
                solution1.sort()
                toAppend.append(solution1)
            else:
                toAppend.append(None)
            if solution2:
                solution2.sort()
                toAppend.append(solution2)
            else:
                toAppend.append(None)
            toAppend.append(aVector)

            testResults.append(toAppend)

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
                if testResults[i][0] != None and 1 in testResults[i][2]:
                    for j in range(i+1, len(equationsList)):
                        if newIndeces[j] == -1:
                            if testResults[i][0] == testResults[j][0] and testResults[i][1] == testResults[j][1] and testResults[i][2] == testResults[j][2]:
                                newIndeces[j] = ind
                                newEquationCount[-1] += equationsCount[j]
                                newEquations[equationsList[j]] = ind

        print("took: ", time.clock()-startedAt,"s")
        print("Reduced from ", len(equationsList)," equations to ", len(newEquationsList), " equations")
        return newEquations
