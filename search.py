import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re

# Function to extract show details from the title
def extract_show_details(title):
    """ Extract structured information from show titles. """
    pattern = re.compile(r"(.+?)\sS(\d+)E(\d+)", re.IGNORECASE)
    match = pattern.search(title)
    if match:
        return match.group(1).strip(), int(match.group(2)), int(match.group(3))
    return title, None, None

# Load data from the SQLite database
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM zipfiles"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Preprocess data and extract features
def preprocess_and_vectorize(df):
    # Extract details
    df['show_details'] = df['name'].apply(extract_show_details)
    df['show_name'], df['season'], df['episode'] = zip(*df['show_details'])

    # Vectorization of the show name
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df['show_name'])

    # Save the vectorizer for later use in the application
    joblib.dump(vectorizer, 'vectorizer.pkl')
    return X

# Main execution function
if __name__ == "__main__":
    db_path = 'Eng Subtitles Database.db'  # Path to your database file

    # Load and preprocess data
    df = load_data(db_path)
    X = preprocess_and_vectorize(df)

    # Save the matrix and dataframe for use in the application
    joblib.dump(X, 'tfidf_matrix.pkl')
    df.to_pickle('processed_data.pkl')  # Save DataFrame with show details
    print("Training completed and data saved.")
