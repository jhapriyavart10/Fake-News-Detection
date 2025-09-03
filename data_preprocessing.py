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
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def preprocess(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"Error: '{input_csv}' not found. Please ensure the dataset is in the current directory.")
        return
    df = pd.read_csv(input_csv)
    if 'label' not in df.columns:
        print("Error: 'label' column not found in the input CSV. Please ensure your dataset has a 'label' column (0=Fake, 1=Real).")
        print("Columns found:", df.columns.tolist())
        return
    df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)
    # Save only relevant columns plus content
    columns_to_save = [col for col in ['id', 'title', 'author', 'text', 'label'] if col in df.columns] + ['content']
    df.to_csv(output_csv, index=False, columns=columns_to_save)

if __name__ == "__main__":
    preprocess('train.csv', 'train_clean.csv')
