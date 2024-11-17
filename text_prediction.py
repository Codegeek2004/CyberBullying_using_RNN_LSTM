# -*- coding: utf-8 -*-
"""Text_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1I_vdvVEYpX5appETb0yLMfhvhhNZfKGt
"""

from google.colab import files
uploaded = files.upload()
# rfc model loaded

#uploaded = files.upload()
# lr model loaded

#vectorizer file imported
uploaded = files.upload()

!pip install contractions
!pip install emoji

import re
import contractions
import emoji
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = contractions.fix(text)  # Expanding contractions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Removing URLs
    text = re.sub(r'\@\w+|\#','', text)  # Removing mentions and hashtags
    text = emoji.demojize(text)  # Converting emojis to text
    text = re.sub(r'_', ' ', text) # Removing underscore from the description of the emojis
    text = re.sub(r'[^\w\s]', '', text)  # Removing special characters
    text = re.sub(r'\d+', '', text)  # Removing digits
    text = word_tokenize(text)
    text = ' '.join([word for word in text if word not in stop_words])  # Removing stopwords
    text = ''.join([lemmatizer.lemmatize(word) for word in text])  # Lemmatization
    return text

# text = "I'm so tired of this! 😡 Check this out: https://example.com #frustrated"
# text = "Go back to where you came from, you're not welcome here."
# text = "You’re such a failure, it’s embarrassing to watch."
# text = "I don’t agree with your opinion, but I respect it."

import pickle

# Open and load the model
with open("rfc.pkl", "rb") as file:
    loaded_model = pickle.load(file)

with open('vectorizer.pkl','rb') as file:
  loaded_vectorizer=pickle.load(file)

print("Model loaded successfully:", type(loaded_model))
print("Model loaded successfully:", type(loaded_vectorizer))

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

text = input("Enter The Comment")
print(predict_cyberbullying(text))