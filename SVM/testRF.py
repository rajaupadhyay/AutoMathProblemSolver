import json
from encoder_functions import *
from global_params import *
from random import shuffle
from checkSolution import *
import RF
import SVM

fp = open('./data/data.json')
data = json.load(fp)

fp = open('./data/equations.json')
equations = json.load(fp)

for iteration in range(10):
	print('Iteration ', iteration)
	twentyFive = int(0.25*len(data))

	shuffle(data)
	test = data[:twentyFive]
	train = data[twentyFive:]

	for i in range(10):
		print('-- SubIteration ', i)
		predict = RF.train(train, (i + 1) * 100)
		predict_SVM = SVM.train(train, 10 ** (i - 4))

		right = 0
		right_SVM = 0

		for datapoint in test:
			predicted = predict(datapoint['question'], False)
			predicted_SVM = predict_SVM(datapoint['question'], False)
			if checkSolution(predicted, datapoint['answers']):
				right += 1
			if checkSolution(predicted_SVM, datapoint['answers']):
				right_SVM += 1
					
		print('RF: ', right/twentyFive, '\t SVM: ', right_SVM/twentyFive)
