import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained('saved_model/')
    tokenizer = DistilBertTokenizer.from_pretrained('saved_model/')
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        label = 'Real' if probs[1] > probs[0] else 'Fake'
        confidence = probs.max()
    return label, confidence

st.title("Fake News Detection with DistilBERT")
model, tokenizer = load_model()

headline = st.text_input("Enter a news headline:")
if st.button("Predict"):
    if headline:
        label, conf = predict(headline, model, tokenizer)
        st.write(f"Prediction: **{label}** (Confidence: {conf:.2f})")

st.markdown("---")
st.write("Batch Prediction (CSV with 'title' column):")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'title' in df.columns:
        results = [predict(t, model, tokenizer) for t in df['title']]
        df['prediction'], df['confidence'] = zip(*results)
        st.write(df[['title', 'prediction', 'confidence']])
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Results", csv, "predictions.csv", "text/csv")
    else:
        st.error("CSV must have a 'title' column.")
