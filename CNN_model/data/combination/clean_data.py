import pandas as pd
import json
import re

json_object_1_f = open('CNN_model/data/combination/data.json')
json_object_1 = json.load(json_object_1_f)
json_object_1_f.close()

questions = []
equations = []
operations = []
solutions = []

dct = {'*': 'Multiplication', '+': 'Addition', '-': 'Subtraction', '/': 'Division'}

for qstn_obj in json_object_1:
    qstn = qstn_obj['question']
    noOfEquations = qstn_obj['noEquations']
    equation = qstn_obj['equations'][0]
    soln = qstn_obj['answers'][0]

    if noOfEquations == 1:
        # nums = re.compile(r"[+-]?\d+(?:\.\d+)?")
        # quants_in_equation = nums.search(equation).group(0)
        quants_in_equation = re.findall(r"\d+(?:\.\d+)?", equation)
        if len(quants_in_equation) == 2:
            operator_in_equation = re.findall(r'[-+\/*]', equation)
            if len(operator_in_equation) == 1:
                qstn = qstn.split()
                qstn = ' '.join(qstn)
                questions.append(qstn)
                equations.append(equation)
                solutions.append(soln)
                operations.append(dct[operator_in_equation[0]])


df = pd.DataFrame()
df['question'] = pd.Series(questions)
df['operation'] = pd.Series(operations)

df.to_csv('combined.csv', sep=',')
