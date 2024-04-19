import sqlite3
import zipfile
import io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from scipy.sparse import vstack  # Import vstack to concatenate sparse matrices

# Connect to SQLite DB


def connect_to_database(db_path):
    return sqlite3.connect(db_path)

# Decode from Latin-1 encoding and decompress


def decode_and_decompress(binary_data):
    with io.BytesIO(binary_data) as f:
        with zipfile.ZipFile(f, 'r') as zip_file:
            with zip_file.open(zip_file.namelist()[0]) as file:
                content = file.read()
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    text = content.decode('latin-1')
    return text

# Load data from the database


def load_data_from_db(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name, content FROM zipfiles")
    rows = cursor.fetchall()
    data = {'text': [], 'label': []}
    for name, content in rows:
        try:
            latin1_decoded = content.decode('latin-1')
            text = decode_and_decompress(bytearray(latin1_decoded, 'latin-1'))
            data['text'].append(text)
            data['label'].append(name)
        except Exception as e:
            print(f"Failed to process data for {name}: {e}")
    return pd.DataFrame(data)


def iter_batches(dataframe, batch_size=1000):
    total_size = len(dataframe)
    for start_idx in range(0, total_size, batch_size):
        end_idx = min(start_idx + batch_size, total_size)
        batch = dataframe[start_idx:end_idx]
        yield batch['text'], batch['label']


# Database path
db_path = 'Eng Subtitles Database.db'
conn = connect_to_database(db_path)

# Extract and prepare data
df = load_data_from_db(conn)

# Setup batch processing
tfidf = TfidfVectorizer(max_features=10000)
batch_generator = iter_batches(df)
all_X_batches = []
all_labels = []

for i, (texts, labels) in enumerate(batch_generator):
    print(f"Vectorizing batch {i + 1}")
    if i == 0:
        X_batch = tfidf.fit_transform(texts)
    else:
        X_batch = tfidf.transform(texts)
    all_X_batches.append(X_batch)
    all_labels.extend(labels)  # Append labels to a list
    print("Batch processed, size:", X_batch.shape)

# Aggregate all batches
# Use vstack to stack sparse matrices vertically
X_aggregated = vstack(all_X_batches)
# Convert label list to pandas Series (or numpy array)
y_aggregated = pd.Series(all_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_aggregated, y_aggregated, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save the model and the vectorizer
joblib.dump(model, 'subtitle_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
