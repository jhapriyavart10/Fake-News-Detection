import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train_clean.csv')
print("Columns in train_clean.csv:", df.columns.tolist())
if 'label' not in df.columns:
    raise ValueError("Column 'label' not found in train_clean.csv. Please check your CSV headers.")

X_train, X_val, y_train, y_val = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_val_tfidf)

print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Baseline')
plt.savefig('baseline_confusion_matrix.png')
plt.close()
