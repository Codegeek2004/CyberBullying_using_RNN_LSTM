# -*- coding: utf-8 -*-
"""Preprocessing .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1q4anW7FRAqzNaKUvE4-OZpVFx6HD5PGb
"""

from google.colab import files

uploaded = files.upload()

import pandas as pd
#df = pd.read_csv('comments_1000_automated.csv',encoding='latin-1')
#df = pd.read_csv('comments_2000_automated.csv',encoding='latin-1')
df = pd.read_csv('comments(1000).csv',encoding='latin-1')
#df = pd.read_csv('comments(2000).csv',encoding='latin-1')
print(df.head())

df.info()

print(df['classification'])

df3 = pd.DataFrame(data={'classification': df['classification']})
print(df3.head())

count_per_column = df3.apply(pd.Series.value_counts).fillna(0)  # fill NaN with 0 if any values are missing
print(count_per_column)

df3.to_csv('classification.csv',index=False)

df2 = pd.DataFrame(data={'comments': df['text'], 'new_comments': df['text']})
df2.head()

"""The first pre-processing step we will do is transform all comments into lower case and create a new column new_commemts."""

df['new_comments'] = df['text'].astype(str).apply(lambda x: " ".join(x.lower() for x in x.split()))
df2['new_comments']=df['new_comments']
df2.head()

"""lets remove contractions.from index 4 comment we have a contraction. ex:it's is it is and he'd is he would"""

!pip install contractions

import contractions
df['new_comments']=df['new_comments'].apply(lambda x:contractions.fix(x))
df2['new_comments']=df['new_comments']
df2.head(50)

"""Lets remove punctuation marks"""

df['new_comments'] = df['new_comments'].str.replace(r'[^\w\s]', '', regex=True)
df2['new_comments']=df['new_comments']
df2.head(50)

"""From above output line 7 has the comment as "He is disgusting with a disgusting ideology. ðŸ¤®".So there are still some characters left so lets encode them and find out what they are and clean the data."""

# Replace unencodable characters with a placeholder
df['new_comments'] = df['new_comments'].apply(lambda x: x.encode('latin1', errors='replace').decode('utf-8', errors='replace'))
df2['new_comments'] = df['new_comments']
df2.head(20)

"""As it is replacing the unwanted character with  '�' this one we will further replace this with an empty string using replace function"""

df['new_comments'] = df['new_comments'].apply(lambda x: x.encode('latin1', errors='replace').decode('utf-8', errors='replace').replace('�', '') )
df2['new_comments'] = df['new_comments']
df2.head(8)

"""As this is not working lets use regex to remove this"""

df['new_comments'] = df['new_comments'].str.replace(r'[^\w\s]', '', regex=True)
df2['new_comments']=df['new_comments']
df2.head(50)

"""Successfully we removed all the special characters and unwanted characters but there are still some numbers lets remove them too"""

df['new_comments']=df['new_comments'].apply(lambda x:''.join([i for i in x if not i.isdigit()]))
df2['new_comments']=df['new_comments']
df2.head(50)

!pip install nltk

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df['new_comments']=df['new_comments'].apply(lambda x:' '.join(x for x in x.split() if x not in stop_words))
df2['new_comments']=df['new_comments']
df2.head(50)

"""lemmatization"""

pip install langdetect

import spacy
from langdetect import detect,LangDetectException
# Load the English language model
nlp = spacy.load("en_core_web_sm")

def lemm(comment):
  c = nlp(comment)
  return ' '.join(token.lemma_ for token in c)

df['new_comments']=df['new_comments'].apply(lambda x: lemm(x))
df2['new_comments']=df['new_comments']
df2.head(50)

df2.head(50)

df2.to_csv('preprocessed.csv',index=False)

from sklearn.model_selection import train_test_split
x=df['new_comments']
y=df['classification']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
X_train.head()