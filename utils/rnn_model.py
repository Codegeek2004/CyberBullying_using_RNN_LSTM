# -*- coding: utf-8 -*-

# Import required libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('preprocessed_data.csv', encoding='latin1')  # Update file path if necessary
df.head()

# Assuming the CSV has columns 'text' and 'classification'
texts = df['text'].astype(str).values
labels = df['classification'].values

# Preprocess labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # Converts labels to 0/1 if they aren't already

# Tokenize and pad the text data (PRE-PADDING)
max_words = 10000  # Maximum number of words in the vocabulary
max_len = 100      # Maximum length of sequences (padding/truncating)

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
pre_padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='pre', truncating='post')

# Split data into training and testing sets
X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(pre_padded_sequences, labels, test_size=0.25, random_state=42)

# Build the RNN model
rnn_pre = Sequential([
    Embedding(input_dim=max_words, output_dim=32),  # Embedding layer
    SimpleRNN(32, activation='relu', return_sequences=False),  # RNN layer
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
rnn_pre.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = rnn_pre.fit(
    X_train_pre, y_train_pre,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_pre, y_test_pre)
)

# Evaluate the model
loss, accuracy = rnn_pre.evaluate(X_test_pre, y_test_pre)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Get predictions from the trained model
y_pred_pre = rnn_pre.predict(X_test_pre)

# Convert predictions to binary class labels
y_pred_pre_class = (y_pred_pre > 0.5).astype('int32')

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test_pre, y_pred_pre_class)
class_names = ['Class 0', 'Class 1']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model in the native Keras format
rnn_pre.save('rnn_model_pre_padding.keras')

# ===== COMMENTED OUT POST-PADDING CODE ===== #
"""
# Tokenize and pad the text data (POST-PADDING)
post_padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Split data into training and testing sets
X_train_post, X_test_post, y_train_post, y_test_post = train_test_split(post_padded_sequences, labels, test_size=0.25, random_state=42)

# Build the RNN model
rnn_post = Sequential([
    Embedding(input_dim=max_words, output_dim=32),  # Embedding layer
    SimpleRNN(32, activation='relu', return_sequences=False),  # RNN layer
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
rnn_post.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = rnn_post.fit(
    X_train_post, y_train_post,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_post, y_test_post)
)

# Evaluate the model
loss, accuracy = rnn_post.evaluate(X_test_post, y_test_post)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Get predictions from the trained model
y_pred_post = rnn_post.predict(X_test_post)

# Convert predictions to binary class labels
y_pred_post_class = (y_pred_post > 0.5).astype('int32')

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test_post, y_pred_post_class)
class_names = ['Class 0', 'Class 1']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
"""
