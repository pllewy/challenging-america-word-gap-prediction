import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd

def predict_next_word(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(sequence)
    print("prediction: ", predicted)

    return predicted

def load_file(file_path: str, sample_size: int = 1000, part='left') -> list:
    word_set = []
    # new_column_names = ['ID', 'source', 'random_name', 'metric_1', 'metric_2', 'metric_3', 'left', 'right']
    x_df = pd.read_csv(file_path + "/in.tsv.xz", delimiter='\t', compression='xz', nrows=sample_size)

    if part == 'left':
        word_set = x_df.iloc[:, -2].tolist()
    else:
        word_set = x_df.iloc[:, -1].tolist()

    # new_column_names = ['target']
    y_df = pd.read_csv(file_path + "/expected.tsv", delimiter='\t', nrows=sample_size)
    y_df = y_df.iloc[:, -1]

    for i in range(len(word_set)):
        word_set[i] = str(word_set[i]).split()
        word_set[i] = word_set[i][-3:]
        word_set[i].append(y_df[i])
        word_set[i] = ' '.join(word_set[i])

    if part == 'right':
        new_sentences = []
        for words in word_set:
            words = words.split()
            reversed_words = words[::-1]

            first_element = reversed_words.pop(0)  # Remove the first element
            reversed_words.append(first_element)

            reversed_sentence = ' '.join(reversed_words)
            new_sentences.append(reversed_sentence)

        word_set = new_sentences

    return word_set

if __name__ == '__main__':

    sentences = load_file('./train')
    right_sentences = load_file('./train', part='right')

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    total_words = len(tokenizer.word_index) + 1

    # Generate input sequences
    input_sequences = []
    for line in sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences and create predictors and label
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = to_categorical(label, num_classes=total_words)

    # Create the model
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(predictors, label, epochs=5, verbose=2) # TODO Change epochs to 100

    # Function to predict the next word

    ####################################

    new_sentences = []
    for words in right_sentences:
        words = words.split()
        reversed_words = words[::-1]
        reversed_sentence = ' '.join(reversed_words)
        new_sentences.append(reversed_sentence)

    # Tokenization
    right_tokenizer = Tokenizer()
    right_tokenizer.fit_on_texts(new_sentences)
    right_total_words = len(right_tokenizer.word_index) + 1

    # Generate input sequences
    input_sequences = []
    for line in new_sentences:
        token_list = right_tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences and create predictors and label
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = to_categorical(label, num_classes=right_total_words)

    # Create the model
    right_model = Sequential()
    right_model.add(Embedding(right_total_words, 10, input_length=max_sequence_len - 1))
    right_model.add(LSTM(100))
    right_model.add(Dense(right_total_words, activation='softmax'))

    right_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    right_model.fit(predictors, label, epochs=5, verbose=2) # TODO Change epochs to 100



    # Predict the next word
    seed_text = "the cat sat"
    pred_right = predict_next_word(right_model, right_tokenizer, seed_text)

    predicted = np.argmax(pred_right, axis=-1)
    pred_word = right_tokenizer.index_word[predicted[0]]

    print(f"Next word prediction: {pred_word}")

    pred_left = predict_next_word(model, tokenizer, seed_text)

    predicted_left = np.argmax(pred_left, axis=-1)
    pred_word_2 = tokenizer.index_word[predicted_left[0]]
    print(f"Next word prediction: {pred_word_2}")

    # final_pred = (pred_right + pred_left) / 2

