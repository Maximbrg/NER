from transformers import AutoTokenizer
from utilits import utilis
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_metric
import numpy as np
import pickle
import argparse


def run(data: tuple = None, model_checkpoint: str = None, task: str = 'ner', batch_size: int = 16):

    def tokenize_and_align_labels(examples):
        label_all_tokens = True
        tokenized_inputs = tokenizer(examples["tokens"], padding=True, max_length=512, truncation=True, is_split_into_words=True)

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

    tokens_s = data[0]
    ner_tags = data[1]

    label_list, label_encoding_dict = utilis.create_label_list(ner_tags=ner_tags)

    # tokens_s, ner_tags = utilis.create_tags()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    train_dataset, test_dataset = utilis.get_un_token_dataset(tokens_s, ner_tags)

    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list),
                                                            ignore_mismatched_sizes=True)

    args = TrainingArguments(
        f"test-{task}",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=1e-5,
        remove_unused_columns=False
    )

    train_tokenized_datasets = train_tokenized_datasets.remove_columns("tokens")
    test_tokenized_datasets = test_tokenized_datasets.remove_columns("tokens")
    train_tokenized_datasets = train_tokenized_datasets.remove_columns("ner_tags")
    test_tokenized_datasets = test_tokenized_datasets.remove_columns("ner_tags")

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
        return {"precision": results["overall_precision"], "recall": results["overall_recall"],
                "f1": results["overall_f1"],
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


if __name__ == "__main__":


    with open('C:\\Users\\max_b\\PycharmProjects\\NER\\data\\2018-4-15.pickle', 'rb') as f:
        data = pickle.load(f)
    run(data=data, model_checkpoint='avichr/heBERT_NER')
