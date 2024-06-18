import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

from preprocessing import load_file


def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(sequence)
    return predicted


def prepare_sequences(sentences, tokenizer):
    input_sequences = []
    for line in sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = to_categorical(label, num_classes=len(tokenizer.word_index) + 1)
    return predictors, label, max_sequence_len

def create_and_train_model(sentences, epochs=5):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    predictors, label, max_sequence_len = prepare_sequences(sentences, tokenizer)
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 10, input_length=max_sequence_len - 1))
    model.add(LSTM(100))
    model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(predictors, label, epochs=epochs, verbose=2)
    # model=""
    # tokenizer=""
    # max_sequence_len=""
    return model, tokenizer, max_sequence_len

if __name__ == '__main__':
    sentences = load_file('./train')
    right_sentences = load_file('./train', part='right')

    model, tokenizer, max_sequence_len = create_and_train_model(sentences, epochs=5)
    right_model, right_tokenizer, right_max_sequence_len = create_and_train_model(right_sentences, epochs=5)

    seed_text = "the cat sat"
    pred_right = predict_next_word(right_model, right_tokenizer, seed_text, right_max_sequence_len)
    predicted_right = np.argmax(pred_right, axis=-1)
    pred_word_right = right_tokenizer.index_word[predicted_right[0]]
    print(f"Next word prediction (right model): {pred_word_right}")

    pred_left = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
    predicted_left = np.argmax(pred_left, axis=-1)
    pred_word_left = tokenizer.index_word[predicted_left[0]]
    print(f"Next word prediction (left model): {pred_word_left}")

