from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from pathlib import Path

def getColumnsFromCsv(pathtofile: str, columns_to_extract: list):
    '''
    pathtofile should be relative
    columns_to_extract is case sensitive
    For example getColumnsFromCsv('data/target_org.csv', ['DATA', 'target'])
    '''
    try:
        extracted_data = []

        # Define the chunk size (adjust this based on your available memory)
        chunk_size = 10000  # You can increase or decrease this value

        # Use a generator to read the CSV file in chunks
        for chunk in pd.read_csv(pathtofile, usecols=columns_to_extract, chunksize=chunk_size, nrows=10000):
            extracted_data.append(chunk)

        # Concatenate the chunks into a single DataFrame
        result = pd.concat(extracted_data, ignore_index=True)

        return result['DATA'].astype(str), result['target'].values
    except FileNotFoundError as e:
        print('No file in this dir\n', e)


# Read data
X_train, y_train = getColumnsFromCsv('case2_part1_train/train_org.csv', ['DATA', 'target'])

# Build vocabulary
t = Tokenizer(num_words=10000)
t.fit_on_texts(X_train)

# Tokenization
X_train = t.texts_to_sequences(X_train)

# Define the desired sequence length
max_sequence_length = your_max_sequence_length  # Replace with your desired value

# Manually pad sequences
X_train_padded = []
for sequence in X_train:
    if len(sequence) < max_sequence_length:
        # Pad with zeros at the beginning to make the sequence the desired length
        padded_sequence = [0] * (max_sequence_length - len(sequence)) + sequence
    else:
        # If the sequence is longer, truncate it to the desired length
        padded_sequence = sequence[:max_sequence_length]
    X_train_padded.append(padded_sequence)

# Convert the list to a NumPy array
X_train_padded = np.array(X_train_padded)

# Build a model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=10000))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fit the model
hist = model.fit(X_train_padded, y_train, epochs=5, batch_size=64)