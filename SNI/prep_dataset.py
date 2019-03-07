import pandas as pd
import re

train_ds = pd.read_csv('SNI/train.csv', sep=',', encoding = "ISO-8859-1")
train_questions = list(train_ds['question'].values)
train_equations = list(train_ds['equation'].values)

train_ds_tuples = list(zip(train_questions, train_equations))


phrase_with_number = []
significant_label = []


for qstn, equation in train_ds_tuples:
    question = qstn.splitlines()
    question = ' '.join(question)
    quantities = re.findall('(num\d+)|\d+', question)

    quantities_in_eqtn = re.findall('(num\d+)', equation)

    question_tokens = question.split()
    total_len = len(question_tokens)

    for qnty in quantities:
        if not qnty:
            continue

        idx = None
        try:
            idx = question_tokens.index(qnty)
        except:
            for tkn in range(len(question_tokens)):
                if qnty in question_tokens[tkn]:
                    idx = tkn
                    break


        # get the words around the number (window of size 3)
        left_index = min(filter(lambda x: x>=0, [idx, idx-1, idx-2, idx-3]))
        right_index = max(filter(lambda x: x<total_len, [idx, idx+1, idx+2, idx+3]))
        window_string = ' '.join(question_tokens[left_index:idx]) + ' ' + ' '.join(question_tokens[idx+1:right_index+1])

        phrase_with_number.append(window_string)

        if qnty in quantities_in_eqtn:
            significant_label.append(1)
        else:
            significant_label.append(0)


outputDF = pd.DataFrame()
outputDF['phrase'] = pd.Series(phrase_with_number)
outputDF['sig_label'] = pd.Series(significant_label)

outputDF.to_csv('sni_dataset.csv', sep=',')
