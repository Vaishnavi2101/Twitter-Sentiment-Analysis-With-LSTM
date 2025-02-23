from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_lstm_model(vocab_size, maxlen=56):
    model = Sequential()
    
    # Embedding Layer
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=maxlen))
    
    # Bidirectional LSTM Layers
    model.add(Bidirectional(LSTM(128, kernel_regularizer=l2(0.1), return_sequences=True, recurrent_regularizer=l2(0.1))))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Fully Connected Layer
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(4, activation='softmax'))
    
    return model
