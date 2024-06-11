import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.preprocessing import LabelEncoder


def load_file(file_path: str, sample_size: int = 3, file_type: str = "input") -> pd.DataFrame:
    if file_type == "input":
        df = pd.read_csv(file_path, delimiter='\t', compression='xz', nrows=sample_size)
        new_column_names = ['ID', 'source', 'random_name', 'metric_1', 'metric_2', 'metric_3', 'left', 'right']
    else:
        df = pd.read_csv(file_path, delimiter='\t', nrows=sample_size)
        new_column_names = ['target']

    df.columns = new_column_names
    return df


if __name__ == '__main__':
    sample_size = 100

    train_df = load_file('./train/in.tsv.xz', sample_size, "input")
    train_df['target'] = load_file('./train/expected.tsv', sample_size, "output")['target']

    x_train = (train_df['left'] + ' [MASK] ' + train_df['right']).astype(str)
    y_train = train_df['target']

    dev_df = load_file('./dev-0/in.tsv.xz', sample_size, "input")
    dev_df['target'] = load_file('./dev-0/expected.tsv', sample_size, "output")['target']

    x_dev = (dev_df['left'] + ' [MASK] ' + dev_df['right']).astype(str)
    y_dev = dev_df['target']

    single_row = train_df.iloc[0]

    for column in single_row.index:
        print(f'{column}: {single_row[column]}')

    # Preprocessing danych
    max_words = 10000
    max_len = 100

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(x_train)
    x_train_sequences = tokenizer.texts_to_sequences(x_train)
    x_dev_sequences = tokenizer.texts_to_sequences(x_dev)
    x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(x_train_sequences, maxlen=max_len)
    x_dev_padded = tf.keras.preprocessing.sequence.pad_sequences(x_dev_sequences, maxlen=max_len)

    # Połącz y_train i y_dev, aby upewnić się, że LabelEncoder widzi wszystkie unikalne etykiety
    all_labels = pd.concat([y_train, y_dev])
    all_labels = all_labels.astype(str)

    # Przekształć etykiety do postaci numerycznej
    encoder = LabelEncoder()
    encoder.fit(all_labels)  # Dopasowanie do wszystkich etykiet
    y_train_encoded = encoder.transform(y_train.astype(str))
    y_dev_encoded = encoder.transform(y_dev.astype(str))
    y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes=len(encoder.classes_))
    y_dev_categorical = tf.keras.utils.to_categorical(y_dev_encoded, num_classes=len(encoder.classes_))

    # Liczba klas
    num_classes = len(encoder.classes_)

    # Model LSTM do przodu
    model_forward = Sequential()
    model_forward.add(Embedding(max_words, 128, input_length=max_len))
    model_forward.add(LSTM(128))
    model_forward.add(Dense(num_classes, activation='softmax'))

    print(x_train_padded.shape, y_train_categorical.shape, x_dev_padded.shape, y_dev_categorical.shape)

    model_forward.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_forward.fit(x_train_padded, y_train_categorical, validation_data=(x_dev_padded, y_dev_categorical), epochs=30, batch_size=32)

    # Model LSTM do tyłu (odwróć kolejność sekwencji)
    x_train_reverse = np.flip(x_train_padded, axis=1)
    x_dev_reverse = np.flip(x_dev_padded, axis=1)

    model_backward = Sequential()
    model_backward.add(Embedding(max_words, 128, input_length=max_len))
    model_backward.add(LSTM(128))
    model_backward.add(Dense(num_classes, activation='softmax'))

    model_backward.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_backward.fit(x_train_reverse, y_train_categorical, validation_data=(x_dev_reverse, y_dev_categorical), epochs=30, batch_size=32)

    # Agregacja predykcji (średnia arytmetyczna)
    preds_forward = model_forward.predict(x_dev_padded)
    preds_backward = model_backward.predict(x_dev_reverse)

    ensemble_preds = (preds_forward + preds_backward) / 2
    ensemble_labels = np.argmax(ensemble_preds, axis=1)
    true_labels = np.argmax(y_dev_categorical, axis=1)

    # Ocena wyników
    accuracy = np.sum(ensemble_labels == true_labels) / len(true_labels)
    print(f"Ensemble Accuracy: {accuracy}")
