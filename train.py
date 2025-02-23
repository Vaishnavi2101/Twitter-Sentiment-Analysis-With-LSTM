import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
from model import create_model

# Load datasets
df_train = pd.read_csv('twitter_training.csv')
df_train.columns = ['ID', 'Company', 'Sentiment', 'Tweet']
df_train.drop(columns=['ID', 'Company'], inplace=True)

# Drop rows where 'Tweet' is NaN
df_train.dropna(subset=['Tweet'], inplace=True)

# Drop duplicate tweets
df_train.drop_duplicates(subset=['Tweet'], inplace=True)

# Convert all values in 'Tweet' column to strings
df_train['Tweet'] = df_train['Tweet'].astype(str)

# Debug: Check for the literal string 'nan' (result of converting NaN to string)
nan_strings = df_train[df_train['Tweet'].str.lower() == 'nan']
print("Rows with 'nan' string:\n", nan_strings)

# Replace 'nan' strings with empty strings
df_train['Tweet'] = df_train['Tweet'].replace('nan', '', regex=False)

# Debug: Print first few rows before cleaning
print("Sample Tweets before cleaning:\n", df_train['Tweet'].head())

# Function to clean tweets
def clean_tweet(text):
    if not isinstance(text, str):  
        print(f"Non-string value encountered: {text}")  # Debugging statement
        return ""  # Replace problematic values with an empty string
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply cleaning function
df_train['Tweet'] = df_train['Tweet'].apply(clean_tweet)

# Debug: Print first few cleaned tweets
print("Sample Tweets after cleaning:\n", df_train['Tweet'].head())

# Encode labels
label_encoder = LabelEncoder()
df_train['Sentiment'] = label_encoder.fit_transform(df_train['Sentiment'])
joblib.dump(label_encoder, 'label_encoder.pkl')

# Tokenization
tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
tokenizer.fit_on_texts(df_train['Tweet'])
joblib.dump(tokenizer, 'tokenizer.pkl')

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df_train['Tweet'])
padded_sequences = pad_sequences(sequences, maxlen=56, padding='post')

# Split data
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, df_train['Sentiment'], test_size=0.2, random_state=42)

# Create model
model = create_model(len(tokenizer.word_index) + 1)

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save model
model.save('sentiment_model.h5')

print("Training complete. Model saved as sentiment_model.h5")