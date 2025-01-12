import os
import pandas as pd

def load_data_from_directory(directory, label):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, encoding="utf-8") as f:
                content = f.read().strip()
                data.append((content, label))
    return data

train_pos_dir = r"aclImdb/train/pos"
train_neg_dir = r"aclImdb/train/neg"

test_pos_dir = r"aclImdb/test/pos"
test_neg_dir = r"aclImdb/test/neg"

train_data = load_data_from_directory(train_pos_dir, 1) + load_data_from_directory(train_neg_dir, 0)
test_data = load_data_from_directory(test_pos_dir, 1) + load_data_from_directory(test_neg_dir, 0)

train_df = pd.DataFrame(train_data, columns=["review", "sentiment"])
test_df = pd.DataFrame(test_data, columns=["review", "sentiment"])

all_data = pd.concat([train_df, test_df], ignore_index=True)

all_data.to_csv("IMDB Dataset.csv", index=False, encoding="utf-8")
print("IMDB Dataset.csv is successfully generated！")
