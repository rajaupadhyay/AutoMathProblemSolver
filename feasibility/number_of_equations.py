import json

fp = open('../data/combined.json')
data = json.load(fp)

number = [0 for i in range(5)]

for entry in data:
    i = entry['noUnknowns']
    if i>3:
        i = 4
    number[i] += 1

print("Number Of Unkowns")
for i in range(4):
    print(i, ": ", number[i])
print(">3 : ", number[4])
