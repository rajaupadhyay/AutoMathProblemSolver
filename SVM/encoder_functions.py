from word2number import w2n
import re
import scipy

ZERO = 0.01

def removeEmptiesAndPunctuation(words):
    words = [word for word in words if word]

    for i in range(len(words)):
        if words[i][-1] in "?!.,;:":
            words[i] = words[i][:-1]

    return words


def findNumbersInWords(words):
    index = 0
    numbers = []
    for i in range(len(words)):
        word = words[i]
        s = word
        j = i

        prevNum = None

        try:
            num = w2n.word_to_num(s)
        except:
            num = None

        while(num != prevNum):
            prevNum = num
            j += 1
            try:
                s += " " + words[j]
                num = w2n.word_to_num(s)
            except:
                num = prevNum

        if num != None:
            words[i] = "a" + str(index)
            for k in range(i+1,j):
                words[k] = ""
            numbers.append(num)
            index += 1

    return numbers

def createTemplate(numbers, equations, unknowns):
    unknown = unknowns[0]
    equationTemplate = equations[0].replace(' ', '').replace(unknown, 'x')
    numbersInEquation = re.findall(r'-?\d+\.?\d*', equationTemplate)
    for num in numbersInEquation:
        if num[0] == '-':
            numbersInEquation.append(num[1:])
    for i in range(len(numbers)):
        number = numbers[i]
        for num in numbersInEquation:
            if number-float(num) < ZERO and number-float(num) > -ZERO:
                equationTemplate = equationTemplate.replace(num, 'a'+str(i))
    return equationTemplate



def replaceNumbers(words, equations, unknowns):
    numbers = findNumbersInWords(words)
    words = removeEmptiesAndPunctuation(words)
    equationTemplate = createTemplate(numbers, equations, unknowns)
    return (words, equationTemplate)


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
    vocab[" # "] = i
    return vocab


def encode(data, vocab):
    y = [d[1] for d in data]
    x = []
    for d in data:
        entry = [0 for l in range(len(vocab))]
        for word in d[0]:
            entry[vocab[word]] += 1
        x.append(entry)
    x = scipy.sparse.csr_matrix(x)
    return (x,y)

def encodeTest(words, vocab):
    entry = [0 for l in range(len(vocab))]
    for word in words:
        try:
            entry[vocab[word]] += 1
        except:
            entry[vocab[" # "]] += 1
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
