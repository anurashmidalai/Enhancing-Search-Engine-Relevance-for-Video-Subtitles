import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the models and resources
model = joblib.load('subtitle_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
vectorizer_shows = joblib.load('vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
df_shows = pd.read_pickle('processed_data.pkl')

def predict_show(subtitle_text):
    """Predict the show based on subtitle text."""
    vectorized_text = vectorizer.transform([subtitle_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

def find_subtitle(show_query):
    """Find the best matching subtitle file name for a given show query."""
    query_vec = vectorizer_shows.transform([show_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    most_similar_idx = similarities.argmax()
    return df_shows.iloc[most_similar_idx]['name']

# Streamlit application interface
st.title('Subtitle Show Predictor and Finder')
st.subheader("Choose the function:")
app_mode = st.selectbox("Select Mode", ["Predict Show from Dialogue", "Find Subtitle File from Show Name"])

if app_mode == "Predict Show from Dialogue":
    user_input = st.text_area("Enter subtitle text here:", height=150)
    if st.button('Predict Show'):
        if user_input:
            show_name = predict_show(user_input)
            st.write(f"This subtitle is likely from: **{show_name}**")
        else:
            st.error("Please enter some subtitle text to predict.")

elif app_mode == "Find Subtitle File from Show Name":
    user_input = st.text_input("Enter show name, e.g., 'Nikita S01 E02'")
    if st.button('Find Subtitle File'):
        if user_input:
            best_match = find_subtitle(user_input)
            st.write(f"Best matching subtitle file: **{best_match}**")
        else:
            st.error("Please enter a show name to search for.")
