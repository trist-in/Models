from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

#unfinetuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)
model.eval()

data_path = "TestReviews.csv"
review_column = "review"
label_column = "class"

df = pd.read_csv(data_path)

def tokenizer_func(text):
    return tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)

tokenized_inputs = [tokenizer_func(text) for text in df[review_column]]

predictions = []
for inputs in tokenized_inputs:
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        predictions.append(predicted_class)
        #print(predicted_class)
        #print(logits)

actual_labels = df[label_column].tolist()
accuracy = accuracy_score(actual_labels, predictions)

print(f"Unfinetuned Model Accuracy: {accuracy:.4f}")
