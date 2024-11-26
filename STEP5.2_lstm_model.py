# -*- coding: utf-8 -*-
"""LSTM Model"""

# Install necessary libraries (if not already installed)
# pip install pandas spacy numpy tensorflow scikit-learn matplotlib seaborn imbalanced-learn

import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load data (Ensure the CSV file is in the working directory or provide full path)
df = pd.read_csv('preprocessed_data.csv', encoding='latin1')
df.head()

# Assuming the CSV has columns 'new_comments' and 'classification'
texts = df['text'].astype(str).values
labels = df['classification'].values

# Step 3: Preprocess labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # Converts labels to 0/1 if they aren't already

# Parameters
vocab_size = 10000  # Adjust this depending on your dataset
max_len = 100      # Maximum length of each input sequence

# **PRE - PADDING**
X_encoded = [one_hot(words, vocab_size) for words in df['text']]  # Encoding the text
X_padded_pre = pad_sequences(X_encoded, padding='pre', maxlen=max_len)  # Padding sequences

# Train-test split
X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X_padded_pre, labels, test_size=0.2, random_state=42)

lstm_pre = Sequential()
lstm_pre.add(Embedding(input_dim=vocab_size, output_dim=128))
lstm_pre.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
lstm_pre.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
lstm_pre.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = lstm_pre.fit(
    X_train_pre,
    y_train_pre,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_pre, y_test_pre)
)

# Evaluate the model
y_pred_pre_probs = lstm_pre.predict(X_test_pre)
y_pred_pre = (y_pred_pre_probs > 0.5).astype(int).flatten()  # Threshold at 0.5 for binary classification

# Accuracy
accuracy = accuracy_score(y_test_pre, y_pred_pre)
print(f"Test Accuracy: {accuracy:.5f}")

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test_pre, y_pred_pre)
class_names = ['Class 0', 'Class 1']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test_pre, y_pred_pre, target_names=class_names))

# **PADDING - POST (COMMENTED OUT)**

# Uncomment the following block if you want to test post-padding

# X_encoded = [one_hot(words, vocab_size) for words in df['text']]  # Encoding the text
# X_padded_post = pad_sequences(X_encoded, padding='post', maxlen=max_len)  # Padding sequences

# # Train-test split
# X_train_post, X_test_post, y_train_post, y_test_post = train_test_split(X_padded_post, labels, test_size=0.3, random_state=42)

# lstm_post = Sequential()
# lstm_post.add(Embedding(input_dim=vocab_size, output_dim=128))
# lstm_post.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
# lstm_post.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# # Compile the model
# lstm_post.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# # Train the model
# history = lstm_post.fit(
#     X_train_post,
#     y_train_post,
#     epochs=10,
#     batch_size=64,
#     validation_data=(X_test_post, y_test_post)
# )

# # Evaluate the model
# y_pred_post_probs = lstm_post.predict(X_test_post)
# y_pred_post = (y_pred_post_probs > 0.5).astype(int).flatten()  # Threshold at 0.5 for binary classification

# # Accuracy
# accuracy = accuracy_score(y_test_post, y_pred_post)
# print(f"Test Accuracy: {accuracy:.5f}")

# # Confusion matrix and classification report
# conf_matrix = confusion_matrix(y_test_post, y_pred_post)
# class_names = ['Class 0', 'Class 1']

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

# print(classification_report(y_test_post, y_pred_post, target_names=class_names))
