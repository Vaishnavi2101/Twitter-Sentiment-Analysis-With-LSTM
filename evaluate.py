# evaluate.py
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load test dataset
df_test = pd.read_csv("twitter_validation.csv")
df_test.columns = ['ID', 'Company', 'Sentiment', 'Tweet']
df_test.drop(columns=['ID', 'Company'], inplace=True)
df_test.dropna(inplace=True)

# Load pre-trained tokenizer and label encoder
tokenizer = joblib.load('tokenizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

df_test['Tweet'] = df_test['Tweet'].apply(clean_tweet)

# Convert text to sequences
test_sequences = tokenizer.texts_to_sequences(df_test['Tweet'])
test_padded = pad_sequences(test_sequences, maxlen=56, padding='post')

# Load model
model = load_model('sentiment_model.h5')

# Evaluate model
loss, accuracy = model.evaluate(test_padded, label_encoder.transform(df_test['Sentiment']))
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
