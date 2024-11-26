# -*- coding: utf-8 -*-

import pandas as pd
import contractions
import nltk
from nltk.corpus import stopwords
import spacy
from langdetect import detect, LangDetectException
from sklearn.model_selection import train_test_split

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Load the SpaCy model, ensure it is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Install it using: python -m spacy download en_core_web_sm")
    exit()

# Read CSV file
try:
    df = pd.read_csv('comments(1000).csv', encoding='latin-1')
except FileNotFoundError:
    print("File 'comments(1000).csv' not found. Please check the file path.")
    exit()

# Ensure required columns exist
if 'classification' not in df.columns or 'text' not in df.columns:
    print("Required columns 'classification' and 'text' not found in the dataset.")
    exit()

# Create classification DataFrame and save count
df3 = pd.DataFrame(data={'classification': df['classification']})
count_per_column = df3.apply(pd.Series.value_counts).fillna(0)
#df3.to_csv('classification.csv', index=False)

# Preprocess the text
df2 = pd.DataFrame(data={'comments': df['text'], 'new_comments': df['text']})

# Convert to lowercase
df['new_comments'] = df['text'].astype(str).apply(lambda x: " ".join(x.lower() for x in x.split()))
df2['new_comments'] = df['new_comments']

# Expand contractions
df['new_comments'] = df['new_comments'].apply(lambda x: contractions.fix(x))
df2['new_comments'] = df['new_comments']

# Remove punctuation
df['new_comments'] = df['new_comments'].str.replace(r'[^\w\s]', '', regex=True)
df2['new_comments'] = df['new_comments']

# Handle encoding issues
df['new_comments'] = df['new_comments'].apply(lambda x: x.encode('latin1', errors='replace').decode('utf-8', errors='replace'))
df2['new_comments'] = df['new_comments']

# Remove unexpected characters
df['new_comments'] = df['new_comments'].apply(lambda x: x.replace('�', ''))
df2['new_comments'] = df['new_comments']

# Remove digits
df['new_comments'] = df['new_comments'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
df2['new_comments'] = df['new_comments']

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['new_comments'] = df['new_comments'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
df2['new_comments'] = df['new_comments']

# Lemmatize text
def lemm(comment):
    c = nlp(comment)
    return ' '.join(token.lemma_ for token in c)

df['new_comments'] = df['new_comments'].apply(lambda x: lemm(x))
df2['new_comments'] = df['new_comments']

# Save preprocessed data (optional)
# df2.to_csv('preprocessed.csv', index=False)

# Split dataset into training and testing sets
x = df['new_comments']
y = df['classification']

# Ensure no missing values
if x.isnull().any() or y.isnull().any():
    print("Null values detected in the dataset. Handling them...")
    x = x.fillna("")
    y = y.fillna("")

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print("Preprocessing and splitting completed successfully.")
