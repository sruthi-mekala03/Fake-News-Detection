import streamlit as st
import pandas as pd
import joblib
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# clean text
def preprocess(text):
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# LOAD trained model (FAST)
model, vectorizer = joblib.load("model.pkl")

st.title("Fake News Detector")

news = st.text_area("Enter news text")

if st.button("Check"):
    clean = preprocess(news)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    prob = model.predict_proba(vector).max()

    result = "FAKE" if prediction == 0 else "REAL"
    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {prob:.2%}")

    # save history
    row = pd.DataFrame([[news, result, prob]],
                       columns=["text","prediction","confidence"])

    try:
        old = pd.read_csv("history.csv")
        row = pd.concat([old,row])
    except:
        pass

    row.to_csv("history.csv", index=False)

    st.subheader("Recent Predictions")
    st.dataframe(row.tail(5))