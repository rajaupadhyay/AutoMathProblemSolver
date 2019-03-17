from global_params import *

def checkSolution(eq, answers):
    correct = False
    if eq:
        for answer in answers:
            possibleUnknowns = "xyz"
            permutations = [[[0]],[[0,1],[1,0]],[[0,1,2],[1,0,2],[0,2,1],[1,2,0],[2,1,0],[2,0,1]]]
            for permutation in permutations[len(answer)-1]:
                tempEq = eq
                for i in range(len(answer)):
                    tempEq = tempEq.replace(possibleUnknowns[permutation[i]], str(answer[i]))
                tempEq = tempEq.split(';')
                correctTemp = True
                for teq in tempEq:
                    try:
                        sol = eval('('+ teq.replace('=', ')-(') +')')
                        if sol > ZERO or sol < -ZERO:
                            correctTemp = False
                    except:
                        correctTemp = False

                correct = correct or correctTemp
                if correct:
                    break
            if correct:
                break

    return correct
