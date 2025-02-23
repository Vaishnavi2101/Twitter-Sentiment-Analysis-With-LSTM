import numpy as np
import pandas as pd
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load test dataset
df_test = pd.read_csv('twitter_validation.csv')
df_test.columns = ['ID', 'Company', 'Sentiment', 'Tweet']
df_test.drop(columns=['ID', 'Company'], inplace=True)

# Load tokenizer & model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
model = load_model("sentiment_model.h5")

# Text Cleaning Function
def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

df_test['Tweet'] = df_test['Tweet'].apply(clean_tweet)

# Encode Labels
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(df_test['Sentiment'])

# Convert texts to sequences
test_sequences = tokenizer.texts_to_sequences(df_test['Tweet'])
test_padded = pad_sequences(test_sequences, maxlen=56, padding='post')

# Evaluate Model
test_loss, test_accuracy = model.evaluate(test_padded, test_labels_encoded)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
