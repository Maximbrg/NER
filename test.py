import hebrew_tokenizer as ht
import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch

import random

random_number = random.uniform(1, 6)
print(random_number)

tokens_s = []
ner_tags = []

with open('./verdict/1.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        tokens = ht.tokenize(line)
        token_s = []
        ner_tag = []
        # tokenize returns a generator!
        for grp, token, token_num, (start_index, end_index) in tokens:
            # print('{}, {}'.format(grp, token))
            token_s.append(token)
            random_number = random.uniform(1, 6)

            if random_number > 4:
                ner_tag.append('O')
            else:
                ner_tag.append('P')

        tokens_s.append(token_s)
        ner_tags.append(ner_tag)

print(tokens_s[1])
print(ner_tags[1])

label_list = ['O', 'P']
label_encoding_dict = {'O': 0, 'P': 1}

task = "ner"
model_checkpoint = "avichr/heBERT_NER"
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def get_un_token_dataset(tokens_s, ner_tags):
    train_df = pd.DataFrame({'tokens': tokens_s, 'ner_tags': ner_tags})
    test_df = pd.DataFrame({'tokens': tokens_s, 'ner_tags': ner_tags})
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return (train_dataset, test_dataset)


train_dataset, test_dataset = get_un_token_dataset(tokens_s, ner_tags)

def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)


model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), ignore_mismatched_sizes=True)

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=1e-5,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model('un-ner.model')