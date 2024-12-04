# Install required packages
!pip install pandas spacy numpy tensorflow scikit-learn matplotlib seaborn imbalanced-learn --quiet

# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dropout

# Load dataset
df = pd.read_csv('classified_comments.csv', encoding='latin1')
df.head()

# Preprocess data
df['new_comments'] = df['new_comments'].astype(str)  # Ensure comments are strings
labels = df['sentiment'].values  # Assuming 'sentiment' column contains target labels

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # Convert labels to numeric format (e.g., 0/1)

# Parameters
vocab_size = 10000  # Vocabulary size
max_len = 100       # Maximum length of input sequences

# Encode text to sequences
X_encoded = tokenizer.texts_to_sequences(df['new_comments'])  # Convert text to integer sequences
X_padded_pre = pad_sequences(X_encoded, padding='pre', maxlen=max_len)  # Padding sequences

# Save the tokenizer for future use
with open('tokenizer_lstm.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

# Convert text to sequences and pad them (pre-padding)
X_encoded = tokenizer.texts_to_sequences(df['new_comments'])
X_padded_pre = pad_sequences(X_encoded, padding='pre', maxlen=max_len)

# Train-test split
X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X_padded_pre, labels, test_size=0.25, random_state=42)

# Define LSTM model with pre-padding
lstm_pre = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
lstm_pre.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history_pre = lstm_pre.fit(
    X_train_pre, y_train_pre,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_pre, y_test_pre)
)

# Evaluate the model
y_pred_pre_probs = lstm_pre.predict(X_test_pre)
y_pred_pre = (y_pred_pre_probs > 0.5).astype(int).flatten()

# Accuracy
accuracy_pre = accuracy_score(y_test_pre, y_pred_pre)
print(f"Test Accuracy (Pre-Padding): {accuracy_pre:.5f}")

# Confusion matrix and classification report
conf_matrix_pre = confusion_matrix(y_test_pre, y_pred_pre)
class_names = ['Class 0', 'Class 1']

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_pre, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Pre-Padding)')
plt.show()

print(classification_report(y_test_pre, y_pred_pre, target_names=class_names))

'''

# Padding sequences (post-padding)
X_padded_post = pad_sequences(X_encoded, padding='post', maxlen=max_len)

# Train-test split
X_train_post, X_test_post, y_train_post, y_test_post = train_test_split(X_padded_post, labels, test_size=0.25, random_state=42)

# Define LSTM model with post-padding
lstm_post = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
lstm_post.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history_post = lstm_post.fit(
    X_train_post, y_train_post,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_post, y_test_post)
)

# Evaluate the model
y_pred_post_probs = lstm_post.predict(X_test_post)
y_pred_post = (y_pred_post_probs > 0.5).astype(int).flatten()

# Accuracy
accuracy_post = accuracy_score(y_test_post, y_pred_post)
print(f"Test Accuracy (Post-Padding): {accuracy_post:.5f}")

# Confusion matrix and classification report
conf_matrix_post = confusion_matrix(y_test_post, y_pred_post)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_post, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Post-Padding)')
plt.show()

print(classification_report(y_test_post, y_pred_post, target_names=class_names))
'''