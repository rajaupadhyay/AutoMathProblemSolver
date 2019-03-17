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
unknowns = []

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
        quants_in_qstn = re.findall(r"\d+(?:\.\d+)?", qstn)
        if len(quants_in_equation) == 2 and len(quants_in_qstn) == 2:
            operator_in_equation = re.findall(r'[-+\/*]', equation)
            if len(operator_in_equation) == 1:
                unknownInQuestion = re.findall(r"[a-zA-Z]+", equation)
                if len(unknownInQuestion) > 1:
                    continue

                qstn = qstn.split()
                qstn = ' '.join(qstn)
                questions.append(qstn)
                equations.append(equation)
                unknowns.append(unknownInQuestion[0])
                solutions.append(soln[0])
                operations.append(dct[operator_in_equation[0]])


df = pd.DataFrame()
df['question'] = pd.Series(questions)
# df['noUnknowns'] = pd.Series([1 for _ in range(len(questions))])
# df['unknowns'] = pd.Series(unknowns)
# df['noEquations'] = pd.Series([1 for _ in range(len(questions))])
df['equations'] = pd.Series(equations)
df['operation'] = pd.Series(operations)
df['answers'] = pd.Series(solutions)

df.to_csv('onlySingleOpDolphin.csv', sep=',')


# out = df.to_json(orient='records', lines=True)
#
# with open('combinedWithAllColumns.json', 'w') as f:
#     f.write(out)
