# Fake News Detection with Transformers

This project helps you detect whether a news headline or article is **Fake** or **Real** using machine learning and deep learning. You will learn how to clean data, train models, and use a web app to make predictions.

### What Am I Building?
- A system that reads news headlines/articles and predicts if they are fake or real.
- It uses two approaches: a simple machine learning model and a powerful deep learning model called DistilBERT.
- You can interact with the system using a web app (Streamlit).

### Dataset Used

For this project, you can use any open-source fake news dataset.  
If you can't find the original Kaggle dataset, you can use the [synthetic Fake News Detection dataset](https://github.com/jhapriyavart10/Fake-News-Detection) you found on Kaggle.

**Dataset Structure:**
- **Rows:** 20,000 news articles
- **Columns:** 7  
  - `title`: Headline or short title  
  - `text`: Main body of the news article  
  - `date`: Publication date  
  - `source`: Media organization (may have missing values)  
  - `author`: Author name (may have missing values)  
  - `category`: Article category (Politics, Technology, Health, Sports, etc.)  
  - `label`: Target classification (`real` or `fake`)

**Note:**  
- The dataset is balanced and realistic.
- About 5% of `source` and `author` values are missing, which is normal.

### Steps to Run the Project

#### 1. Get the Data
- Download the dataset CSV file from Kaggle (use the synthetic dataset you found).
- Put the CSV file (e.g., `fake_news_dataset.csv`) in your project folder.
- If the label column uses text (`real`/`fake`), you may need to convert it to numbers (`1` for real, `0` for fake) in your preprocessing script.

#### 2. Install Required Libraries
- Open your terminal or command prompt.
- Run this command to install all necessary Python libraries:
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn nltk torch transformers streamlit
  ```

#### 3. Clean the Data
- Run the data preprocessing script:
  ```bash
  python data_preprocessing.py
  ```
- This script combines the title and text of each news article, removes unnecessary words and symbols, and saves the cleaned data as `train_clean.csv`.

#### 4. Train a Simple Model (Baseline)
- Run the baseline model script:
  ```bash
  python baseline_model.py
  ```
- This uses a basic machine learning method (TF-IDF + Logistic Regression) to predict fake/real news.
- You will see results like accuracy and a confusion matrix (shows correct and incorrect predictions).

#### 5. Train a Deep Learning Model (DistilBERT)
- Run the transformer training script:
  ```bash
  python train_transformer.py
  ```
- This uses DistilBERT, a modern deep learning model for text, to learn from your data.
- The best model is saved for later use.

#### 6. Use the Web App
- Start the Streamlit app:
  ```bash
  streamlit run app.py
  ```
- You will see a web page where you can:
  - Type a news headline and get a prediction (Fake/Real) with a confidence score.
  - Upload a CSV file with headlines for batch predictions.

### How Does the Code Work?
- **Data Preprocessing:** Cleans and prepares the text so models can learn better.
- **Baseline Model:** Uses simple math to find patterns in words and predict fake/real.
- **Transformer Model:** Uses advanced deep learning to understand the meaning of text.
- **Streamlit App:** Lets you use the models easily through a web interface.

### Common Issues and Solutions
- **Missing Columns:** Make sure your CSV file has `title`, `text`, and `label` columns. If your label is text, update the preprocessing script to convert it to numbers.
- **NLTK Errors:** The code will download necessary data automatically.
- **GPU/CPU:** The code uses your computer's GPU if available, otherwise it uses CPU.

### Tips for Beginners
- Run each script one by one and check the outputs.
- If you see an error, read the message carefully—it usually tells you what is wrong.
- You can change the code to try different models or datasets.

### Useful Links
- [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/your-dataset-link)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

If you have questions or want to learn more, feel free to explore the code and try new things!


