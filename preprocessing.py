import pandas as pd
import re

def load_file(file_path: str, sample_size: int = 1000, part='left') -> list:
    word_set = []
    x_df = pd.read_csv(file_path + "/in.tsv.xz", delimiter='\t', compression='xz', nrows=sample_size)
    y_df = pd.read_csv(file_path + "/expected.tsv", delimiter='\t', nrows=sample_size).iloc[:, -1]

    if part == 'left':
        word_set = x_df.iloc[:, -2].tolist()
    else:
        word_set = x_df.iloc[:, -1].tolist()

    word_set = [preprocess_text(str(words)) for words in word_set]

    for i in range(len(word_set)):
        word_set[i] = str(word_set[i]).split()
        word_set[i] = word_set[i][-3:]
        word_set[i].append(y_df[i])
        word_set[i] = ' '.join(word_set[i])
        print(f"word_set[{i}]: {word_set[i]}")

    if part == 'right':
        word_set = [' '.join(words.split()[::-1]) for words in word_set]

    return word_set

def preprocess_text(text):
    text = text.replace('\n', ' ')
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = re.sub(r'\[\d+–\d+\]|\[\d+(,\d+)*\]', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\sąćęłńóśźżĄĆĘŁŃÓŚŹŻáéíóúüñÁÉÍÓÚÜÑ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    return text