import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load datasets
df = pd.read_csv('preprocessed(1000).csv')
df2 = pd.read_csv('classification(1000).csv')

# Combine and clean datasets
new_df = pd.DataFrame({'new_comments': df['new_comments'], 'classification': df2['classification']})
df3 = new_df.dropna(subset=['new_comments', 'classification'])

# Split data into features and labels
x = df3['new_comments']
y = df3['classification']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Save vectorizer for future use
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Utility function to plot confusion matrix
def plot_confusion_matrix(y_actual, y_pred, labels, title):
    cm = confusion_matrix(y_actual, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel('Actual', fontsize=13)
    plt.title(title, fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()
    plt.show()

# Logistic Regression
lr = LogisticRegression(max_iter=1000, solver='lbfgs')
lr.fit(X_train_vec, y_train)
y_pred_lr = lr.predict(X_test_vec)
print("\nLogistic Regression Performance:")
print("Accuracy: ", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
plot_confusion_matrix(y_test, y_pred_lr, ['cyberbullying', 'Not cyberbullying'], 'Logistic Regression')

# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_vec, y_train)
y_pred_rfc = rfc.predict(X_test_vec)
print("\nRandom Forest Classifier Performance:")
print("Accuracy: ", accuracy_score(y_test, y_pred_rfc))
print(classification_report(y_test, y_pred_rfc))
plot_confusion_matrix(y_test, y_pred_rfc, ['cyberbullying', 'Not cyberbullying'], 'Random Forest Classifier')

# Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train_vec, y_train)
y_pred_mnb = mnb.predict(X_test_vec)
print("\nNaive Bayes Performance:")
print("Accuracy: ", accuracy_score(y_test, y_pred_mnb))
print(classification_report(y_test, y_pred_mnb))
plot_confusion_matrix(y_test, y_pred_mnb, ['cyberbullying', 'Not cyberbullying'], 'Naive Bayes')

# Decision Tree
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train_vec, y_train)
y_pred_dtc = dtc.predict(X_test_vec)
print("\nDecision Tree Performance:")
print("Accuracy: ", accuracy_score(y_test, y_pred_dtc))
print(classification_report(y_test, y_pred_dtc))
plot_confusion_matrix(y_test, y_pred_dtc, ['cyberbullying', 'Not cyberbullying'], 'Decision Tree')

# Support Vector Machine
svc = SVC(kernel='linear', C=1.0, random_state=42)
svc.fit(X_train_vec, y_train)
y_pred_svc = svc.predict(X_test_vec)
print("\nSupport Vector Machine Performance:")
print("Accuracy: ", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
plot_confusion_matrix(y_test, y_pred_svc, ['cyberbullying', 'Not cyberbullying'], 'Support Vector Machine')

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_vec, y_train)
y_pred_knn = knn.predict(X_test_vec)
print("\nK-Nearest Neighbors Performance:")
print("Accuracy: ", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
plot_confusion_matrix(y_test, y_pred_knn, ['cyberbullying', 'Not cyberbullying'], 'K-Nearest Neighbors')

# Save Random Forest model
with open("rfc.pkl", "wb") as file:
    pickle.dump(rfc, file)
print("\nRandom Forest model saved!")

with open("rfc.pkl", "wb") as file:
    pickle.dump(rfc, file)
print("\nRandom Forest model saved!")
