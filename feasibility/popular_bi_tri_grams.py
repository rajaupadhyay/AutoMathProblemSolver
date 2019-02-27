import json

fp = open('../data/combined.json')
data = json.load(fp)

vocab = []
count = []
bi = []
tri = []
biCount = []
triCount = []
reverseVocab = {}
reverseBi = {}
reverseTri = {}

print("Building Vocab")

for datapoint in data:
    text = datapoint['question'].lower()
    sentences = text.split(". ")
    for sentence in sentences:
        words = sentence.split(" ")
        for i in range(len(words)):
            try:
                words[i] = float(words[i])
                words[i] = "NUM"
            except:
                pass
        for i in range(len(words)):
            word = words[i]
            try:
                j = reverseVocab[word]
                count[j] += 1
            except:
                reverseVocab[word] = len(vocab)
                vocab.append(word)
                count.append(1)
            if i < len(words)-1:
                bigram = words[i] + ' ' + words[i+1]
                try:
                    k = reverseBi[bigram]
                    biCount[k] += 1
                except:
                    reverseBi[bigram] = len(bi)
                    biCount.append(1)
                    bi.append(bigram)
            if i < len(words)-2:
                trigram = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
                try:
                    k = reverseTri[trigram]
                    triCount[k] += 1
                except:
                    reverseTri[trigram] = len(tri)
                    triCount.append(1)
                    tri.append(trigram)





# most popular words
mostPopular = sorted(range(len(count)), reverse=True, key=lambda k: count[k])
mostPopularBi = sorted(range(len(biCount)), reverse=True, key=lambda k: biCount[k])
mostPopularTri = sorted(range(len(triCount)), reverse=True, key=lambda k: triCount[k])

N = 50
print("%d most populat words" % N)
for i in range(N):
    print(vocab[mostPopular[i]], count[mostPopular[i]])

print("%d most populat bigrams" % N)
for i in range(N):
    print(bi[mostPopularBi[i]], biCount[mostPopularBi[i]])

print("%d most populat trigrams" % N)
for i in range(N):
    print(tri[mostPopularTri[i]], triCount[mostPopularTri[i]])
