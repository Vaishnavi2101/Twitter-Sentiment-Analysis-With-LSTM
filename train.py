import numpy as np
import pandas as pd
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_lstm_model

# Load datasets
df_train = pd.read_csv('twitter_training.csv')

# Rename and drop unnecessary columns
df_train.columns = ['ID', 'Company', 'Sentiment', 'Tweet']
df_train.drop(columns=['ID', 'Company'], inplace=True)

# Handle missing values by filling NaN Tweets with an empty string
df_train['Tweet'] = df_train['Tweet'].fillna('')

# Text Cleaning Function
def clean_tweet(text):
    if not isinstance(text, str):  # Ensure text is a string
        text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions & hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\d', ' ', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

df_train['Tweet'] = df_train['Tweet'].apply(clean_tweet)

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(df_train['Sentiment'])

# Tokenization
tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
tokenizer.fit_on_texts(df_train['Tweet'])
train_sequences = tokenizer.texts_to_sequences(df_train['Tweet'])
train_padded = pad_sequences(train_sequences, maxlen=56, padding='post')

# Build and Train Model
vocab_size = len(tokenizer.word_index) + 1
model = build_lstm_model(vocab_size)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_padded,
    train_labels_encoded,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)


model.save("sentiment_model.h5")
print("✅ Model saved successfully!")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved successfully!")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("✅ Label encoder saved successfully!")

