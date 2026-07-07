import streamlit as st
import joblib
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ----------------------------
# Download required NLTK data
# ----------------------------
resources = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

for resource, path in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# ----------------------------
# Load saved model and vectorizer
# ----------------------------
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------------------
# Text preprocessing function
# ----------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in string.punctuation and word not in stop_words
    ]

    return " ".join(tokens)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_review = preprocess_text(review)
        vectorized_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_review)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")
