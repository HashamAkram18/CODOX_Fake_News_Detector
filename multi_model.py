# prompt: just give me updated code for streamlit part where i can select diferent model through selection bar, celalring input feilds after each prediction 

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



# Load the saved models and vectorizer
model_paths = {
    "Logistic Regression": "LogisticRegression.joblib",
    "Passive Aggressive Classifier": "PassiveAggressiveClassifier.joblib",
    "Multinomial Naive Bayes": "MultinomialNB.joblib",
    "XGBoost Classifier": "XGBClassifier.joblib",
    "Random Forest Classifier": "RandomForestClassifier.joblib"
}

loaded_models = {}
for name, path in model_paths.items():
    loaded_models[name] = joblib.load(path)

loaded_vectorizer = joblib.load('count_vectorizer.joblib')

# Download NLTK resources if necessary
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_new_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text

def tokenize_and_predict(text, model, vectorizer):
    preprocessed_text = preprocess_new_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_vectorized)
    return prediction

# Streamlit app
st.title("Fake News Detection")

# Model selection
selected_model_name = st.selectbox("Select Model", list(loaded_models.keys()))
selected_model = loaded_models[selected_model_name]

# Initialize session state for text input if not already
if 'text_input' not in st.session_state:
    st.session_state['text_input'] = ''

# Set the value of the text area based on session state
text_input = st.text_area("Enter a news article:", value=st.session_state['text_input'])

# Prediction button
if st.button("Predict"):
    if text_input:
        prediction = tokenize_and_predict(text_input, selected_model, loaded_vectorizer)
        if prediction[0] == 1:
            st.success("The news article is predicted to be REAL.")
        else:
            st.error("The news article is predicted to be FAKE.")
        # Clear the text input field
        st.session_state['text_input'] = ''
    else:
        st.warning("Please enter some text.")
