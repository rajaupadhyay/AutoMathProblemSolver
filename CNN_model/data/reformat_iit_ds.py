import pandas as pd
import random

df = pd.read_csv('CNN_model/data/iit_test.csv', sep=',', encoding = "ISO-8859-1")

reformatted_questions = []
ops = []

for idx, row in df.iterrows():
    qstn = row['question']
    operation = row['operation']

    rand_nums = [random.randint(1,1000) for _ in range(2)]

    while 'num' in qstn and rand_nums:
        qstn = qstn.replace('num', str(rand_nums.pop(0)), 1)

    reformatted_questions.append(qstn)
    ops.append(operation)

res_df = pd.DataFrame()
res_df['question'] = pd.Series(reformatted_questions)
res_df['operation'] = pd.Series(ops)

res_df.to_csv('reformatted_iit.csv', sep=',')
