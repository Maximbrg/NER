import hebrew_tokenizer as ht
import random
import docx2txt
import os
import pickle
import pandas as pd

import tag_rules
import h5py
import pandas


def create_tags(txt: str = None):
    """

    :return:
    """
    tokens_s = []
    ner_tags = []

    lines = txt.splitlines()
    for line in lines:

        tokens = ht.tokenize(line)

        token_s = []
        ner_tag = []
        # tokenize returns a generator!
        for grp, token, token_num, (start_index, end_index) in tokens:
            # print('{}, {}'.format(grp, token))
            token_s.append(token)
            ner_tag.append(tag_rules.money_entity(token))

        if len(token_s) == 0:
            continue
        tokens_s.append(token_s)
        ner_tags.append(ner_tag)

    return tokens_s, ner_tags


# my_text = docx2txt.process("test.docx")
# print(my_text)
jj = 0
path = 'word_data/2018-4-15/'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
tokens = []
tags = []
for i, file in enumerate(files):
    file_path = path + file
    my_text = docx2txt.process(file_path)
    tokens_s, ner_tags = create_tags(my_text)
    tokens.extend(tokens_s)
    tags.extend(ner_tags)
    # for x in ner_tags:
    #     if 'I-Money' in x:
    #         print(jj)
    #         jj += 1
    #         print(i)
    #         print('============')
    # if i == 500:
    #     break

    # print(i)

with open('/data/2018-4-15.pickle', 'wb') as handle:
    pickle.dump((tokens, tags), handle, protocol=pickle.HIGHEST_PROTOCOL)