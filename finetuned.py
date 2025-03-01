from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          TrainingArguments, Trainer)
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data_path = 'TestReviews.csv'
review_column = 'review'
class_column = 'class'

df = pd.read_csv(data_path)
df[class_column] = df[class_column].astype(int)

df_train, df_test = train_test_split(df, test_size=0.2)
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenizer_func(examples):
    return {
        **tokenizer(examples[review_column], truncation=True),
        "labels": examples[class_column],
    }

tokenized_train = train_dataset.map(tokenizer_func, batched=True)
tokenized_test = test_dataset.map(tokenizer_func, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    eval_strategy="steps",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")

model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
