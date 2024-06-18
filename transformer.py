from transformers import pipeline

from preprocessing import load_file

def fill_mask_predictions(sentences, fill_masker, max_length=450):
    predictions = []
    total_length = 0

    for sentence in sentences:
        words = sentence.split()
        total_length += len(words)
        if total_length > max_length:
            break
        if len(words) > max_length - 1:
            words = words[:max_length - 1]
        masked_sentence = ' '.join(words) + ' [MASK]'
        result = fill_masker(masked_sentence)
        predicted_word = result[0]['token_str']
        predictions.append(predicted_word)
    return predictions

if __name__ == '__main__':
    sentences = load_file('./train')
    right_sentences = load_file('./train', part='right')

    fill_masker = pipeline("fill-mask", model="bert-base-uncased")

    left_predictions = fill_mask_predictions(sentences, fill_masker)
    right_predictions = fill_mask_predictions(right_sentences, fill_masker)

    print("Left model predictions:")
    for i, prediction in enumerate(left_predictions):
        print(f"Sentence {i}: {prediction}")

    print("Right model predictions:")
    for i, prediction in enumerate(right_predictions):
        print(f"Sentence {i}: {prediction}")
