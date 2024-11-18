import re
import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess the input text by:
    - Lowercasing
    - Expanding contractions
    - Removing URLs, mentions, hashtags, emojis, special characters, and digits
    - Tokenizing and removing stopwords
    - Lemmatizing the words
    """
    text = text.lower()  # Lowercasing
    text = contractions.fix(text)  # Expanding contractions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Removing URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Removing mentions and hashtags
    text = emoji.demojize(text)  # Converting emojis to text
    text = re.sub(r'_', ' ', text)  # Removing underscores from emojis
    text = re.sub(r'[^\w\s]', '', text)  # Removing special characters
    text = re.sub(r'\d+', '', text)  # Removing digits

    # Tokenize and filter out stopwords
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]  # Removing stopwords
    text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatization
    return ' '.join(text)  # Join the tokens back into a string

# Load the saved model and vectorizer
with open("rfc.pkl", "rb") as file:
    loaded_model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)

print("Model loaded successfully:", type(loaded_model))
print("Vectorizer loaded successfully:", type(loaded_vectorizer))

def predict_cyberbullying(text):
    """
    Preprocess the input text, transform it using the vectorizer,
    and predict whether it is cyberbullying or not.
    """
    cleaned_text = preprocess_text(text)

    # Convert the preprocessed text into the same feature format as the training data
    text_vector = loaded_vectorizer.transform([cleaned_text])  # Using the saved vectorizer

    # Predict using the loaded model
    prediction = loaded_model.predict(text_vector)  # 0 = Not Cyberbullying, 1 = Cyberbullying

    # Return the result as either 'Cyberbullying' or 'Not Cyberbullying'
    return "Cyberbullying" if prediction == 1 else "Not Cyberbullying"

# Get user input and predict
text = input("Enter The Comment: ")
print(predict_cyberbullying(text))
