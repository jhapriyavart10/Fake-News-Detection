import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def preprocess(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"Error: '{input_csv}' not found. Please ensure the dataset is in the current directory.")
        return
    df = pd.read_csv(input_csv)
    if 'label' not in df.columns:
        print("Error: 'label' column not found in the input CSV. Please ensure your dataset has a 'label' column.")
        print("Columns found:", df.columns.tolist())
        return
    # Convert label to numeric if needed
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'real': 1, 'fake': 0})
    df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)
    columns_to_save = [col for col in ['title', 'text', 'date', 'source', 'author', 'category', 'label'] if col in df.columns] + ['content']
    df.to_csv(output_csv, index=False, columns=columns_to_save)

if __name__ == "__main__":
    preprocess('train.csv', 'train_clean.csv')
