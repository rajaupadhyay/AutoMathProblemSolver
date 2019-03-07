import pandas as pd
import re

# idx 66
test_ds = pd.read_csv('CNN_Model/cnn_for_hyde_dataset/test.csv', sep=',', encoding = "ISO-8859-1")

questions = list(test_ds['question'].values)
equations = list(test_ds['equation'].values)

data_tuples = list(zip(questions, equations))

formatted_questions = []
operators = []

op_dict = {'-':'Subtraction', '+':'Addition', '*':'Multiplication', '/':'Division'}

for question, equation in data_tuples:
    question = question.replace('\n', '')

    qstn = re.sub('(num\d+)|\d+', 'num', question)
    operator = re.findall('[+*/-]', equation)[0]

    formatted_questions.append(qstn)
    operators.append(op_dict[operator])

output_df = pd.DataFrame()
output_df['question'] = pd.Series(formatted_questions)
output_df['operation'] = pd.Series(operators)

output_df.to_csv('iit_test.csv', sep=',')
