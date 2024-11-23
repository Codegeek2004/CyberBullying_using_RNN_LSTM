# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
import pickle  # Import pickle

print("TensorFlow version:", tf.__version__)

# Load the dataset
df = pd.read_csv('preprocessed_data.csv', encoding='latin1')

# Extract features and labels
X = df['text']  # Preprocessed text column
y = df['classification'].values  # Labels

# Parameters
vocab_size = 10000  # Vocabulary size for encoding
max_len = 100       # Maximum sequence length

# One-hot encode and pad the sequences
X_encoded = [one_hot(str(text), vocab_size) for text in X]  # Ensure all text is converted to string
X_padded = pad_sequences(X_encoded, padding='pre', maxlen=max_len)  # Pad sequences to uniform length

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.3, random_state=42)

# Build the LSTM model
lstm = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
lstm.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = lstm.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Evaluate the model
y_pred_probs = lstm.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Threshold at 0.5 for binary classification

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.5f}")

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = ['Class 0', 'Class 1']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred, target_names=class_names))

# Save the trained model architecture and weights using pickle
with open("lstm_model.pkl", "wb") as file:
    pickle.dump({
        "model_json": lstm.to_json(),  # Save model architecture
        "model_weights": lstm.get_weights(),  # Save model weights
    }, file)
print("Model saved successfully as 'lstm_model.pkl'.")

# To Load the model:
# ------------------
# with open("lstm_model.pkl", "rb") as file:
#     data = pickle.load(file)
#     loaded_model = tf.keras.models.model_from_json(data["model_json"])
#     loaded_model.set_weights(data["model_weights"])
#     print("Model loaded successfully.")
