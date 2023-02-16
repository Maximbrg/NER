from torch.utils.data import DataLoader
from preprocessing.dataset import CustomNERDataset
import pandas as pd
from datasets import Dataset
import hebrew_tokenizer as ht
import random


def create_label_list(ner_tags: [] = None):
    """

    :param ner_tags:
    :return:
    """

    label_list = []
    label_encoding_dict = {}
    i = 0

    for sentence in ner_tags:
        for y in sentence:

            if y not in label_list:
                label_list.append(y)
                label_encoding_dict[y] = i
                i += 1

    return label_list, label_encoding_dict


def get_un_token_dataset(tokens_s, ner_tags):
    """

    :param tokens_s:
    :param ner_tags:
    :return:
    """
    train_df = pd.DataFrame({'tokens': tokens_s, 'ner_tags': ner_tags})
    test_df = pd.DataFrame({'tokens': tokens_s, 'ner_tags': ner_tags})

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset



