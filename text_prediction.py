import re
import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow as tf

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess the input text."""
    text = text.lower()
    text = contractions.fix(text)  # Expand contractions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = emoji.demojize(text)  # Convert emojis to text
    text = re.sub(r'_', ' ', text)  # Replace underscores with spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = word_tokenize(text)  # Tokenize the text
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize
    return ' '.join(text)

# Load model and handle missing file errors
try:
    loaded_model = tf.keras.models.load_model("lstm.h5")
    print("Model loaded successfully")
except FileNotFoundError:
    raise FileNotFoundError("The model file 'lstm.h5' was not found. Please provide the correct path.")

# Parameters for encoding and padding
vocab_size = 5000
max_length = 100

def predict_cyberbullying(text):
    """Predict whether the text contains cyberbullying."""
    # Preprocess input text
    cleaned_text = preprocess_text(text)
    
    # Encode the text using one-hot encoding
    encoded_text = one_hot(cleaned_text, vocab_size)
    
    # Pad the encoded text
    padded_text = pad_sequences([encoded_text], maxlen=max_length, padding='post')
    
    # Make prediction
    prediction = loaded_model.predict(padded_text)
    
    # Interpret the prediction
    return "Cyberbullying" if prediction[0][0] >= 0.5 else "Not Cyberbullying"

# Example usage
if __name__ == "__main__":
    input_text = "You're such a loser! Nobody likes you."
    result = predict_cyberbullying(input_text)
    print(f"Prediction: {result}")
