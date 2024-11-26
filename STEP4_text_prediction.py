# -*- coding: utf-8 -*-
import re
import pickle
import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK dependencies are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = contractions.fix(text)  # Expanding contractions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Removing URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Removing mentions and hashtags
    text = emoji.demojize(text)  # Converting emojis to text
    text = re.sub(r'_', ' ', text)  # Removing underscore from the description of the emojis
    text = re.sub(r'[^\w\s]', '', text)  # Removing special characters
    text = re.sub(r'\d+', '', text)  # Removing digits
    text = word_tokenize(text)
    text = ' '.join([word for word in text if word not in stop_words])  # Removing stopwords
    text = ''.join([lemmatizer.lemmatize(word) for word in text])  # Lemmatization
    return text

# Load the pre-trained models and vectorizer
try:
    with open("rfc.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        loaded_vectorizer = pickle.load(file)

    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# Prediction function
def predict_cyberbullying(text):
    cleaned_text = preprocess_text(text)

    # Convert the preprocessed text into the same feature format as the training data
    text_vector = loaded_vectorizer.transform([cleaned_text])  # Using the saved vectorizer

    # Predict using the loaded model
    prediction = loaded_model.predict(text_vector)  # 0 = Not Cyberbullying, 1 = Cyberbullying

    if prediction == 1:
        return "Cyberbullying"
    else:
        return "Not Cyberbullying"

# Main function to get user input and make a prediction
if __name__ == "__main__":
    text = input("Enter a comment to analyze: ")
    result = predict_cyberbullying(text)
    print(f"Prediction: {result}")
