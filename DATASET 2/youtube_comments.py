# -*- coding: utf-8 -*-
"""Youtube_Comments.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fC4TyXfz_v_PRe3EJFJ0BK-hvkbKOPse
"""

pip install --upgrade google-api-python-client

import googleapiclient.discovery
import pandas as pd
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
# Get the API key
api_key = os.getenv("DEVELOPER_API_KEY")

print(f"My API key is: {api_key}")

YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

youtube = googleapiclient.discovery.build(
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_API_KEY
)

def get_comments(video):
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video,
        maxResults=100  # Set to 100 for the initial request
    )

    comments = []
    response = request.execute()

    # Get comments from the response
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['textOriginal'],
        ])

    while (1 == 1):
      try:
        nextPageToken = response['nextPageToken']
      except KeyError:
        break
      nextPageToken = response['nextPageToken']
      nextRequest = youtube.commentThreads().list(part = "snippet", videoId = video, maxResults = 100, pageToken = nextPageToken)
      response = nextRequest.execute()
      for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            comment['textOriginal'],
        ])

    df2 = pd.DataFrame(comments, columns=['author', 'text'])
    return df2

df = get_comments('DQzKw30LeTA')
df

df.to_csv('comments.csv')