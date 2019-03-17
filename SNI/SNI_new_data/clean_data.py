import pandas as pd
import re

phrase = []
siglabel = []

content = None
with open('SNI/SNI_new_data/sni_data.txt') as f:
    content = f.readlines()

content = [x.strip() for x in content]

for itx in content:
    split_items = itx.split('||')

    if len(split_items) != 2:
        continue

    qstn = split_items[0]
    eqtn = split_items[1]

    question = qstn.splitlines()
    question = ' '.join(question)
    quantities = re.findall(r"\d+(?:\.\d+)?", question)

    quantities_in_eqtn = re.findall(r"\d+(?:\.\d+)?", eqtn)

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

        phrase.append(window_string)

        if qnty in quantities_in_eqtn:
            siglabel.append(1)
        else:
            siglabel.append(0)


res_df = pd.DataFrame()
res_df['phrase'] = pd.Series(phrase)
res_df['sig_label'] = pd.Series(siglabel)

res_df.to_csv('sni_dataset_just_new.csv', sep=',')
