import json

fp = open('../data/combined.json')
data = json.load(fp)
numRedData = []

count = 0
for datapoint in data:
    q = datapoint['question']
    q = q.lower()

    q = q.replace('thousand', '1000')

    keyHundreds = ['§', 'one hundred', 'two hundred', 'three hundred', 'four hundred', 'five hundred', 'six hundred', 'seven hundred', 'eight hundred', 'nine hundred', 'hundred', 'a hundred']
    keyTens = ['§', '§', 'twenty', 'thirty', 'fourty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    keyTeens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', '§', '§', 'twelfth']
    keyOnes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', '§', 'first', 'second', 'third']
    for hundreds in keyHundreds:
        if hundreds in q:
            figure = keyHundreds.index(hundreds) % 10
            if figure < 1:
                figure = 1
            figure *= 100
            for teens in keyTeens:
                    str1 = [ hundreds + ' ' + teens, hundreds + 'and' + teens]
                    figure1 = figure + 10 + keyTeens.index(teens)
                    for s in str1:
                        q = q.replace(s, str(figure1))
            for tens in keyTens:
                    str1 = [ hundreds + ' ' + tens, hundreds + 'and' + tens]
                    figure1 = figure + 10 * (keyTens.index(tens) % 10)
                    for ones in keyOnes:
                        figure2 = figure1 + (keyOnes.index(ones) % 10)
                        str2 = [s + ' ' + ones for s in str1] + [s + '-' + ones for s in str1] + [s + '' + ones for s in str1]
                        for s in str2:
                            q = q.replace(s, str(figure2))
                    for s in str1:
                        q = q.replace(s, str(figure1))
            for ones in keyOnes:
                if hundreds + ones in q or hundreds + 'and' + ones in q:
                    str1 = [ hundreds + ones, hundreds + 'and' + ones]
                    figure1 = figure + keyOnes.index(ones)%10
                    for s in str1:
                        q = q.replace(s, str(figure1))

            q = q.replace(hundreds, str(figure))

    for teens in keyTeens:
            figure1 = 10 + keyTeens.index(teens)
            q = q.replace(teens, str(figure1))
    for tens in keyTens:
            figure1 = 10 * (keyTens.index(tens) % 10)
            for ones in keyOnes:
                figure2 = figure1 + (keyOnes.index(ones) % 10)
                str2 = [tens + bind + ones for bind in [' ', '-', '']]
                for s in str2:
                    q = q.replace(s, str(figure2))
            q = q.replace(tens, str(figure1))
    for ones in keyOnes:
        figure1 = keyOnes.index(ones) % 10
        q = q.replace(ones, str(figure1))

    datapoint['question'] = q


sent = 0
num = 0

for datapoint in data:
    tempSent = False
    tempNum = False
    sentences = datapoint['question'].replace('$', '').split('. ')
    for sentence in sentences:
        if '?' in sentence:
            continue
        words = sentence.split(' ')
        numbers = []
        for word in words:
            try:
                numbers.append(float(word))
            except:
                x = 0
        if not len(numbers):
            tempSent = True
        else:
            isContained = False
            for number in numbers:
                try:
                    substr = "%d" % number
                    for equation in datapoint['equations']:
                        if substr in equation:
                            isContained = True
                            break
                        if isContained:
                            break
                except:
                    x = 0
            if not isContained:
                tempNum = True
    if tempSent:
        sent += 1
    if tempNum:
        num += 1
        numRedData.append(datapoint)


print(sent, num)

fpout = open('../data/num-red.json', 'w')
json.dump(numRedData, fpout, indent=2)
