import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the saved model and vectorizer
loaded_model = joblib.load('LogisticRegression.joblib')
loaded_vectorizer = joblib.load('count_vectorizer.joblib')

# Download NLTK resources if necessary
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_new_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    text = ' '.join(words)
    return text

def tokenize_and_predict(text, model, vectorizer):
    # Preprocess the new text
    preprocessed_text = preprocess_new_text(text)
    # Transform the preprocessed text using the loaded vectorizer
    text_vectorized = vectorizer.transform([preprocessed_text])
    # Make prediction using the loaded model
    prediction = model.predict(text_vectorized)
    return prediction

# Inference on new article
new_text = input("Enter a news article: ")
prediction = tokenize_and_predict(new_text, loaded_model, loaded_vectorizer)

if prediction[0] == 1:
    print("The news article is predicted to be REAL.")
else:
    print("The news article is predicted to be FAKE.")
