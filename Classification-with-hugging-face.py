
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd

def classify_and_evaluate(model, tokenizer, sentences, labels):
    # Tokenization
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Model Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Compute Softmax
    probs = softmax(logits, dim=-1)

    # Make Predictions
    predictions = torch.argmax(probs, dim=-1)

    # Evaluate
    correct = (predictions == labels).sum().item()
    total = len(sentences)
    accuracy = correct / total * 100

    return f'Accuracy: {accuracy}%'

if __name__ == "__main__":
   

    # Load Pre-trained Model and Tokenizer
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #Prepare Data
    sentences = ["I love programming.", "I hate bugs."]
    labels = torch.tensor([1, 0])  # 1: Positive, 0: Negative

    # Classify and Evaluate
    result = classify_and_evaluate(model, tokenizer, sentences, labels)
    print(result)

# if I wanted to use Imdb labelelled dataset for further training
df = pd.read_csv('IMDB_Dataset.csv.gz', compression='gzip', encoding='utf-8')

