import pandas as pd
import re
import random
import json

json_object_1_f = open('CNN_model/data/MAWPS/SingleOp.json')
json_object_1 = json.load(json_object_1_f)
json_object_1_f.close()

json_object_2_f = open('CNN_model/data/MAWPS/AddSub.json')
json_object_2 = json.load(json_object_2_f)
json_object_2_f.close()

question = []
equation = []
operation = []
solution = []
dct = {'*': 'Multiplication', '+': 'Addition', '-': 'Subtraction', '/': 'Division'}


for qstn_obj in json_object_1:
    numOfObj = qstn_obj['lAlignments']
    if len(numOfObj) == 2:
        qstn = qstn_obj['sQuestion']
        eqtn = qstn_obj['lEquations'][0]
        soln = qstn_obj['lSolutions'][0]

        question.append(qstn)
        equation.append(eqtn)
        solution.append(soln)

        operator_in_equation = re.findall(r'[-+\/*]', eqtn)
        print(operator_in_equation)
        operation.append(dct[operator_in_equation[0]])


for qstn_obj in json_object_1:
    qstn = qstn_obj['sQuestion']
    eqtn = qstn_obj['lEquations'][0]
    soln = qstn_obj['lSolutions'][0]

    quants_in_equation = re.findall(r'\d+.?\d*', eqtn)
    if len(quants_in_equation) == 2:
        operator_in_equation = re.findall(r'[-+\/*]', eqtn)

        question.append(qstn)
        equation.append(eqtn)
        solution.append(soln)

        operator_in_equation = re.findall(r'[-+\/*]', eqtn)
        operation.append(dct[operator_in_equation[0]])



df = pd.DataFrame()
df['question'] = pd.Series(question)
df['equation'] = pd.Series(equation)
df['operation'] = pd.Series(operation)
df['solution'] = pd.Series(solution)

df.to_csv('MAWPS.csv')







# df = pd.read_csv('singleop_ph.csv', encoding = "ISO-8859-1")
#
# ops = []
# dct = {'*': 'Multiplication', '+': 'Addition', '-': 'Subtraction', '/': 'Division'}
# for idx, row in df.iterrows():
#     operator_in_equation = re.findall(r'[-+\/*]', row['equation'])
#     ops.append(dct[operator_in_equation[0]])
#
# df['operation'] = pd.Series(ops)
# df.to_csv('formatted_singleop.csv', sep=',')

# df_1 = pd.read_csv('data/formatted_singleop.csv', encoding = "ISO-8859-1")
# df_2 = pd.read_csv('data/train.csv')
#
# question = []
# operation = []
# equation = []
# solution = []
#
# for idx, row in df_1.iterrows():
#     question.append(row['question'])
#     operation.append(row['operation'])
#     equation.append(row['equation'])
#     solution.append(row['answer'])
#
# for idx, row in df_2.iterrows():
#     question.append(row['question'])
#     operation.append(row['operation'])
#     equation.append(row['equation'])
#     solution.append(row['answer'])
#
# new_test_df = pd.DataFrame()
# new_test_df['question'] = pd.Series(question)
# new_test_df['operation'] = pd.Series(operation)
# new_test_df['equation'] = pd.Series(equation)
# new_test_df['answer'] = pd.Series(solution)
#
# new_test_df.to_csv('new_train.csv', sep=',')



'''
# new_data_set = []
# for question, label in data_set:
#     quantities_from_question = re.findall(r'\d+\.?\d*', question)
#     qstn = question
#     for qnt in quantities_from_question:
#         qstn = qstn.replace(qnt, 'num')
#
#     new_data_set.append((qstn, label))
#
#
# with open('singleop_shuffled_num_replaced.pickle', 'wb') as handle:
#     pickle.dump(new_data_set, handle)
'''


'''
Add training examples
'''

# train_ds = pd.read_csv('data/new_train.csv', sep=',', encoding = "ISO-8859-1")
#
# add = 'What is the sum of {} and {}?'
# sub = 'What is {} subtracted from {}?'
# div = 'What is {} divided by {}?'
# mul = 'What is the product of {} and {}?'
#
#
#
#
# def add_samples(num, qstn_string, type='Addition', op='+'):
#     question = []
#     operation = []
#     equation = []
#     solution = []
#
#     x = [random.randint(1, 100) for _ in range(num)]
#     y = [random.randint(1, 100) for _ in range(num)]
#
#     values = list(zip(x, y))
#
#     for tup in values:
#         qstn = qstn_string.format(tup[0], tup[1])
#         eqtn = 'X={}{}{}'.format(tup[0], op, tup[1])
#         soln = eval('{}{}{}'.format(tup[0], op, tup[1]))
#
#         question.append(qstn)
#         operation.append(type)
#         equation.append(eqtn)
#         solution.append(soln)
#
#     return question, operation, equation, solution
#
#
#
#
# idx = [882+i for i in range(1,41)]
#
# g_question = list(train_ds['question'].values)
# g_operation = list(train_ds['operation'].values)
# g_equation = list(train_ds['equation'].values)
# g_solution = list(train_ds['answer'].values)
#
# question, operation, equation, solution = add_samples(10, add, type='Addition', op='+')
# g_question.extend(question)
# g_operation.extend(operation)
# g_equation.extend(equation)
# g_solution.extend(solution)
#
#
# question, operation, equation, solution = add_samples(10, sub, type='Subtraction', op='-')
# g_question.extend(question)
# g_operation.extend(operation)
# g_equation.extend(equation)
# g_solution.extend(solution)
#
# question, operation, equation, solution = add_samples(10, div, type='Division', op='/')
# g_question.extend(question)
# g_operation.extend(operation)
# g_equation.extend(equation)
# g_solution.extend(solution)
#
# question, operation, equation, solution = add_samples(10, mul, type='Multiplication', op='*')
# g_question.extend(question)
# g_operation.extend(operation)
# g_equation.extend(equation)
# g_solution.extend(solution)
#
# output_df = pd.DataFrame()
# output_df['question'] = pd.Series(g_question)
# output_df['operation'] = pd.Series(g_operation)
# output_df['equation'] = pd.Series(g_equation)
# output_df['answer'] = pd.Series(g_solution)
#
# output_df.to_csv('train_synthetic.csv', sep=',')
