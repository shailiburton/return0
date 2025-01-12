# pip install torch transformers datasets scikit-learn
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split  
import torch  
import pandas as pd  
import numpy as np  
import torch
import transformers


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Read IMDB dataset.csv
data = pd.read_csv("IMDB Dataset.csv")  
print(data.head())  

# Data Cleaning and Preprocessing dataset
data['review'] = data['review'].str.strip()  # 去除文本首尾的空格
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # 将情感转化为0和1
data = data.sample(frac=0.1, random_state=42)  # 仅使用10%的数据

#Divide into review and sentiment
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['review'], data['sentiment'], test_size=0.2, random_state=42
)


from transformers import BertTokenizer, BertForSequenceClassification
model_name = "bert-base-uncased"
# Use bert Tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name,num_labels=2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512, return_tensors="pt")
# truncation
# padding
# return_tensors="pt"

# Define PyTorch Dataset class
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
         item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
         item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
         return item

train_dataset = IMDbDataset(train_encodings, train_labels.tolist())
val_dataset = IMDbDataset(val_encodings, val_labels.tolist())


model.to(device) 

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)  # AdamW optimizer，learning rate=5e-5
loss_fn = torch.nn.CrossEntropyLoss()  

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  
val_loader = DataLoader(val_dataset, batch_size=4) 

def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train() 
    total_loss = 0
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad() 
        batch = {k: v.to(device) for k, v in batch.items()}  
        outputs = model(**batch) 
        loss = loss_fn(outputs.logits, batch['labels']) 
        total_loss += loss.item() 
        loss.backward()  
        optimizer.step()  

        if step % 10 == 0:  
            print(f"Batch {step}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)  


# Define validation function
def evaluate(model, dataloader):
    model.eval()  
    correct = 0
    with torch.no_grad():  
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}  
            outputs = model(**batch)  
            predictions = torch.argmax(outputs.logits, dim=1)  
            correct += (predictions == batch['labels']).sum().item()  
    return correct / len(dataloader.dataset)  

# Training
for epoch in range(1):  
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn)  
    val_accuracy = evaluate(model, val_loader)  
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

# Testing
model.eval()  
text = "The movie was absolutely wonderful!"  
tokens = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)  
prediction = model(**tokens)  
print(f"Sentiment: {'Positive' if torch.argmax(prediction.logits) == 1 else 'Negative'}")  
