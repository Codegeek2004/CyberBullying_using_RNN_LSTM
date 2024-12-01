# Install required packages
!pip install contractions emoji tensorflow nltk

# Import necessary libraries
import re
import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = contractions.fix(text)  # Expanding contractions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Removing URLs
    text = re.sub(r'@\w+|#', '', text)  # Removing mentions and hashtags
    text = emoji.demojize(text)  # Converting emojis to text
    text = re.sub(r'_', ' ', text)  # Removing underscores in emoji descriptions
    text = re.sub(r'[^\w\s]', '', text)  # Removing special characters
    text = re.sub(r'\d+', '', text)  # Removing digits
    text = word_tokenize(text)  # Tokenizing
    text = ' '.join([word for word in text if word not in stop_words])  # Removing stopwords
    text = ' '.join([lemmatizer.lemmatize(word) for word in text])  # Lemmatizing
    return text

# Load the pre-trained model
try:
    model = load_model('lstm_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define constants
vocab_size = 10000  # Vocabulary size for one-hot encoding
max_length = 100  # Maximum length of padded sequences

# Function to predict cyberbullying using the LSTM model
def predict_cyberbullying_lstm(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)

    # One-hot encode the cleaned text
    encoded_text = one_hot(cleaned_text, vocab_size)

    # Pad the sequences to the required length
    padded_text = pad_sequences([encoded_text], maxlen=max_length, padding='pre')

    # Predict using the LSTM model
    prediction = model.predict(padded_text, verbose=0)  # Suppress verbose output

    # Return the prediction
    return "Cyberbullying" if prediction[0][0] > 0.5 else "Not Cyberbullying"

# Main script to get user input and make predictions
if __name__ == "__main__":
    text = input("Enter The Comment: ")
    print(predict_cyberbullying_lstm(text))
