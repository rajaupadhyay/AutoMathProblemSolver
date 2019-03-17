from word2number import w2n
import re
import scipy

MIN_EXAMPLE = 2
ZERO = 0.01
MAX_EQUATION_LENGTH=40
MAX_NO_EQUATIONS=3


def removeEmptiesAndPunctuation(words):
    words = [word for word in words if word]

    for i in range(len(words)):
        if words[i][-1] in "?!.,;:$%":
            words[i] = words[i][:-1]
        if len(words[i]) > 0 and words[i][0] in "$":
            words[i] = words[i][1:]

    return words


def findNumbersInWords(words):
    ind = 0
    numbers = []
    for i in range(len(words)):
        word = words[i]
        s = word
        j = i

        prevNum = None

        couldBeNum=True
        num = None

        for letter in word:
            if letter not in "0123456789.()/*+-":
                couldBeNum = False
                break

        if couldBeNum:
            try:
                num = eval(word)
                num = float(num)
            except:
                num = None

        if num != None:
            if num not in numbers:
                words[i] = "a" + str(ind)
                numbers.append(num)
                ind += 1
            else:
                tempInd = numbers.index(num)
                words[i] = "a" + str(tempInd)
        else:
            try:
                num = w2n.word_to_num(s)
            except:
                num = None

            while(num != prevNum):
                prevNum = num
                j += 1
                try:
                    num = None
                    if words[j] == "point":
                        s += " " + words[j] + " " + words[j+1]
                        tempNum = w2n.word_to_num(s)
                        if tempNum != prevNum:
                            num = tempNum
                            j += 1
                    if num == None:
                        s += " " + words[j]
                        num = w2n.word_to_num(s)
                except:
                    num = prevNum

            if num != None:
                if num not in numbers:
                    words[i] = "a" + str(ind)
                    numbers.append(num)
                    ind += 1
                else:
                    tempInd = numbers.index(num)
                    words[i] = "a" + str(tempInd)
                for k in range(i+1,j):
                    words[k] = ""


    return numbers

def createTemplate(numbers, equations, unknowns):

    possibleUnknowns = "xyz"
    equationTemplate = ''
    for equation in equations:
        equationTemplate += equation
        equationTemplate += ';'
    equationTemplate = equationTemplate[:-1]

    unknowns.sort(reverse=True)
    pre = []
    for char in possibleUnknowns:
        if char in unknowns:
            unknowns.remove(char)
            pre.append(char)
    unknowns = pre + unknowns
    for i in range(len(unknowns)):
        equationTemplate = equationTemplate.replace(unknowns[i], possibleUnknowns[i])

    charset1a = unknowns + [')', 'x']
    charset2b = unknowns + ['(', 'x']
    charset2a = charset1a + [str(i) for i in range(10)] + ['.']
    charset1b = charset2b + [str(i) for i in range(10)] + ['.']
    for a in charset1a:
        for b in charset1b:
            equationTemplate = equationTemplate.replace(a+b, a+'*'+b)
    for a in charset2a:
        for b in charset2b:
            equationTemplate = equationTemplate.replace(a+b, a+'*'+b)

    numbersInEquation = re.findall(r'-?\d*\.?\d*', equationTemplate)
    # numbersInEquation = re.findall(r"\d+(\.\d+)?", equationTemplate)
    numbersInEquation = list(filter(lambda s : s != '' and s != '-' and s != '-0', numbersInEquation))
    numbersInEquation.sort(key=(lambda s : len(s)), reverse=True)

    for num in numbersInEquation:
        if num[0] == '-':
            numbersInEquation.append(num[1:])

    positiveNumbersInEquation = list(filter(lambda s : s[0] != '-', numbersInEquation))
    positiveNumbersInEquation.sort(key=(lambda s : len(s)), reverse=True)

    for i in range(len(positiveNumbersInEquation)):
        num = positiveNumbersInEquation[i]
        equationTemplate = equationTemplate.replace(num, "b" + str(i))
        if "bb" + str(i) in equationTemplate:
            equationTemplate = equationTemplate.replace("bb" + str(i), "b"+str(num))

    for i in range(len(numbers)):
        number = numbers[i]
        for num in numbersInEquation:
            if number-float(num) < ZERO and number-float(num) > -ZERO:
                prefix = ''
                if num[0] == '-':
                    prefix = '-'
                    num = num[1:]
                ind = positiveNumbersInEquation.index(num)
                equationTemplate = equationTemplate.replace(prefix + "b" + str(ind), "a" + str(i))

    for i in range(len(positiveNumbersInEquation)):
        equationTemplate = equationTemplate.replace("b" + str(i), positiveNumbersInEquation[i])

    return equationTemplate



def replaceNumbers(words, equations, unknowns):
    numbers = findNumbersInWords(words)
    words = removeEmptiesAndPunctuation(words)
    equationTemplate = createTemplate(numbers, equations, unknowns)
    return (words, equationTemplate, numbers)


def buildVocab(data):
    vocab = {}
    i = 0
    for d in data:
        words = d[0]
        for word in words:
            try:
                a = vocab[word]
            except:
                vocab[word] = i
                i+= 1
    try:
        a = vocab[" # "]
    except:
        vocab[" # "] = i
    return vocab


def encode(data, vocab):
    y = [d[1] for d in data]
    x = []
    for d in data:
        entry = [0 for l in range(len(vocab))]
        for word in d[0]:
            entry[vocab[word]] = 1
        x.append(entry)
    x = scipy.sparse.csr_matrix(x)
    return (x,y)

def encodeTest(words, vocab):
    entry = [0 for l in range(len(vocab))]
    for word in words:
        try:
            entry[vocab[word]] = 1
        except:
            entry[vocab[" # "]] = 1
    return entry

def addBiTriGrams(words):
    biTri = []
    for i in range(len(words)-2):
        biTri.append(words[i] +" "+ words[i+1])
        biTri.append(words[i] +" "+ words[i+1] +" "+ words[i+2])
    if len(words) >= 2:
        biTri.append(words[-2] +" "+ words[-1])
    words += biTri
    return words

def getRelevantWords(data, minLength):
    tempVocab = {}
    for d in data:
        words = d[0]
        for word in words:
            try:
                tempVocab[word] += 1
            except:
                tempVocab[word] = 1
    for word,count in tempVocab.items():
        if count < minLength:
            tempVocab[word] = False
    return tempVocab

def replaceIrrelevantWords(words, tempVocab):
    for i in range(len(words)):
        try:
            count = tempVocab[words[i]]
            if not count:
                words[i] = " # "
        except:
            words[i] = " # "
    return words

def removeWordsWithLowFreq(data, tempVocab):
    for d in data:
        words = d[0]
        replaceIrrelevantWords(words, tempVocab)
    return data





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
