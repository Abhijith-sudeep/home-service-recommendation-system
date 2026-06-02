import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ----------------------------
# TEXT CLEANING FUNCTION
# ----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # remove punctuation
    text = re.sub(r"\s+", " ", text)             # remove extra spaces
    return text.strip()


# ----------------------------
# LOAD DATASET
# ----------------------------

print("Loading dataset...")

# Make sure train.csv is inside ml folder
data = pd.read_csv("train.csv")

# Keep only required columns
data = data[["comment_text", "toxic"]]

# Drop missing values
data = data.dropna()

# Clean text
data["comment_text"] = data["comment_text"].apply(clean_text)

X = data["comment_text"]
y = data["toxic"]


# ----------------------------
# SPLIT DATA
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# TF-IDF VECTORIZATION
# ----------------------------

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


# ----------------------------
# TRAIN MODEL
# ----------------------------

print("Training model...")

model = MultinomialNB()
model.fit(X_train_vectors, y_train)


# ----------------------------
# EVALUATE MODEL
# ----------------------------

y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")


# ----------------------------
# SAVE MODEL
# ----------------------------

pickle.dump(model, open("toxic_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")