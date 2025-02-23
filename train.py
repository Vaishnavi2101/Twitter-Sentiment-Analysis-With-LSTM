]import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib

# Load datasets
df_train = pd.read_csv('twitter_training.csv')
df_train.columns = ['ID', 'Company', 'Sentiment', 'Tweet']

# Drop unnecessary columns & handle NaNs
df_train.drop(columns=['ID', 'Company'], inplace=True)
df_train.dropna(subset=['Tweet'], inplace=True)  # Drop rows where 'Tweet' is NaN

# Ensure all values in 'Tweet' column are strings
df_train['Tweet'] = df_train['Tweet'].astype(str)

# Remove duplicates
df_train.drop_duplicates(subset=['Tweet'], inplace=True)

# Find non-string values in the 'Tweet' column (debugging step)
non_string_tweets = df_train[~df_train['Tweet'].apply(lambda x: isinstance(x, str))]
if not non_string_tweets.empty:
    print("Non-string values found:", non_string_tweets)

# Tweet cleaning function
def clean_tweet(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply cleaning
df_train['Tweet'] = df_train['Tweet'].apply(clean_tweet)

# Encode labels
label_encoder = LabelEncoder()
df_train['Sentiment'] = label_encoder.fit_transform(df_train['Sentiment'])
joblib.dump(label_encoder, 'label_encoder.pkl')

# Tokenization
tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>", filters='')
tokenizer.fit_on_texts(df_train['Tweet'])
joblib.dump(tokenizer, 'tokenizer.pkl')

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df_train['Tweet'])
padded_sequences = pad_sequences(sequences, maxlen=56, padding='post', truncating='post')

# Split data
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, df_train['Sentiment'], test_size=0.2, random_state=42)

# Load model
from model import create_model
model = create_model(len(tokenizer.word_index) + 1)

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Save model
model.save('sentiment_model.h5')
