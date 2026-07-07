import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load model and vectorizer
@st.cache_resource
def load_resources():
    model = joblib.load("logistic_regression_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    return model, vectorizer, lemmatizer, stop_words

model, tfidf_vectorizer, lemmatizer, stop_words = load_resources()


# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(cleaned_tokens)


# ---------------- Streamlit UI ----------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and click **Analyze Sentiment**.")

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Example: This movie was amazing. I loved the acting!"
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        try:
            # Preprocess
            cleaned_review = preprocess_text(review)

            # Vectorize
            review_vector = tfidf_vectorizer.transform([cleaned_review])

            # Predict
            prediction = model.predict(review_vector)[0]

            # Display result
            if prediction == 1:
                st.success("😊 Positive Sentiment")
            else:
                st.error("😠 Negative Sentiment")

        except Exception as e:
            st.error(f"Error: {e}")
