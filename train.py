# train.py
import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from model import create_model
from tensorflow.keras.models import load_model

# Load training dataset
df_train = pd.read_csv('twitter_training.csv')
df_train.columns = ['ID', 'Company', 'Sentiment', 'Tweet']
df_train.drop(columns=['ID', 'Company'], inplace=True)
df_train.dropna(inplace=True)
df_train.drop_duplicates(subset=['Tweet'], inplace=True)

def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

df_train['Tweet'] = df_train['Tweet'].apply(clean_tweet)
train_texts = df_train['Tweet'].values
train_labels = df_train['Sentiment'].values

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=56, padding='post')

vocab_size = len(tokenizer.word_index) + 1
model = create_model(vocab_size, 56)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_padded, train_labels_encoded, epochs=10, validation_split=0.1)

model.save('sentiment_model.h5')
np.save('label_encoder_classes.npy', label_encoder.classes_)
