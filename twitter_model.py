
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# data from: https://www.kaggle.com/cosmos98/twitter-and-reddit-sentimental-analysis-dataset
raw_data = pd.read_csv(r"C:\Users\Graham\Desktop\TR Data\Twitter_Data.csv")
# this is neccessary because panda sometimes makes things the first column floats when we get to tokenizing.
# its probably not neccesarry to make sure the category column ints, but it can't hurt.
raw_data.clean_text = raw_data.clean_text.astype(str)

np_data = raw_data.to_numpy()
print(np_data[0][1])
comments = []
labels = []
for item in np_data:
    if item[1] == -1:
        labels.append(0)
        comments.append(item[0])
    elif item[1] == 1:
        comments.append(item[0])
        labels.append(1)
training_size = int(len(comments) // 1.5)
print("training size: " + str(training_size))
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(comments)
vocab_size = 10000
embedding_dim = 16
max_length = 25
trunc_type = 'post'
padding_type = 'post'

# splitting into training and testing comments
training_comments = comments[0:training_size]
print("training comments size " + str(len(training_comments)))
testing_comments = comments[training_size:]
print("testing comments size " + str(len(testing_comments)))

# splitting into training and testing sentiment labels
training_labels = labels[0:training_size]
print("training labels size " + str(len(training_labels)))
testing_labels = labels[training_size:]
print("testing labels size " + str(len(testing_labels)))
# create tokenizer and create vocab
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_comments)
# training_sequence padding
training_sequences = tokenizer.texts_to_sequences(training_comments)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# testing sequences padding
testing_sequences = tokenizer.texts_to_sequences(testing_comments)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# apparently it doesn't like normal lists and needs numpy arrays
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
num_epochs = 100
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

