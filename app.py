import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK data (if not already downloaded)
# This might take a moment the first time you run the app locally
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load the saved model and vectorizer
@st.cache_resource
def load_resources():
    model = joblib.load('logistic_regression_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return model, vectorizer, lemmatizer, stop_words

model, tfidf_vectorizer, lemmatizer, stop_words = load_resources()

# Text preprocessing function (must be identical to the one used during training)
def preprocess_text_for_streamlit(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# Streamlit app layout
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")

# Text area for user input
user_input = st.text_area("Enter your movie review here:", height=200)

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the user's review
        cleaned_review = preprocess_text_for_streamlit(user_input)

        # Transform the cleaned review using the loaded TF-IDF vectorizer
        # The vectorizer expects an iterable, so pass [cleaned_review]
        review_vectorized = tfidf_vectorizer.transform([cleaned_review])

        # Make a prediction using the loaded model
        prediction = model.predict(review_vectorized)

        # Display the result
        if prediction[0] == 1:
            st.success("**Sentiment: Positive** ✨")
        else:
            st.error("**Sentiment: Negative** 😠")
    else:
        st.warning("Please enter a movie review to analyze.")
