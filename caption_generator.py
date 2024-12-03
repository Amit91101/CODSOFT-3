from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, RepeatVector, Concatenate
from tensorflow.keras.layers import TimeDistributed, Flatten, Reshape, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
import numpy as np

# Parameters
max_length = 34  # Maximum length of the caption
vocab_size = 10000  # Vocabulary size
embedding_dim = 256
units = 512

# Create the caption generation model
def create_caption_model(vocab_size, max_length, embedding_dim, units):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
