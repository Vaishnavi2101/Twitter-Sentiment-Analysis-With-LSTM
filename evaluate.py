# evaluate.py
import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load test dataset
df_test = pd.read_csv('twitter_validation.csv')
df_test.columns = ['ID', 'Company', 'Sentiment', 'Tweet']
df_test.drop(columns=['ID', 'Company'], inplace=True)
df_test.dropna(inplace=True)

def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

df_test['Tweet'] = df_test['Tweet'].apply(clean_tweet)
test_texts = df_test['Tweet'].values
test_labels = df_test['Sentiment'].values

# Load label encoder and model
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
tokenizer.fit_on_texts(test_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=56, padding='post')

test_labels_encoded = label_encoder.transform(test_labels)
model = load_model('sentiment_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_padded, test_labels_encoded)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
