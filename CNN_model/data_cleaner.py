import pandas as pd
import re

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

df_1 = pd.read_csv('data/formatted_singleop.csv', encoding = "ISO-8859-1")
df_2 = pd.read_csv('data/train.csv')

question = []
operation = []
equation = []
solution = []

for idx, row in df_1.iterrows():
    question.append(row['question'])
    operation.append(row['operation'])
    equation.append(row['equation'])
    solution.append(row['answer'])

for idx, row in df_2.iterrows():
    question.append(row['question'])
    operation.append(row['operation'])
    equation.append(row['equation'])
    solution.append(row['answer'])

new_test_df = pd.DataFrame()
new_test_df['question'] = pd.Series(question)
new_test_df['operation'] = pd.Series(operation)
new_test_df['equation'] = pd.Series(equation)
new_test_df['answer'] = pd.Series(solution)

new_test_df.to_csv('new_train.csv', sep=',')



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
