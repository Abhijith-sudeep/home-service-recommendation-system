import pickle
import os

# Get absolute path to this folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "toxic_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

def check_toxic(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)[0][1]
    return prediction[0] == 1, probability