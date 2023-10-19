import evaluate
import numpy as np
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from evaluate import EvaluationModule
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from transformers.models.bert.modeling_bert import \
    BertForSequenceClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args: TrainingArguments = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
)
metric: EvaluationModule = evaluate.load("accuracy")

ds: DatasetDict = load_dataset("yelp_review_full")

tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizedDS: DatasetDict = ds.map(tokenize_function, batched=True)

small_train_dataset: DatasetDict = (
    tokenizedDS["train"].shuffle(seed=42).select(range(1000))
)
small_eval_dataset: DatasetDict = (
    tokenizedDS["test"].shuffle(seed=42).select(range(1000))
)

model: BertForSequenceClassification = (
    AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=5,
    )
)

trainer: Trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
