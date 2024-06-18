from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from preprocessing import load_file


def fill_mask_predictions(sentences, model, tokenizer, max_length=450):
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

        inputs = tokenizer(masked_sentence, return_tensors="pt")
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        token_logits = model(**inputs).logits
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_token = torch.argmax(mask_token_logits, dim=1)
        predicted_word = tokenizer.decode(top_token)
        predictions.append(predicted_word.strip())
    return predictions


if __name__ == '__main__':
    sentences = load_file('./train')
    right_sentences = load_file('./train', part='right')

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    left_predictions = fill_mask_predictions(sentences, model, tokenizer)
    right_predictions = fill_mask_predictions(right_sentences, model, tokenizer)

    print("Left model predictions:")
    for i, prediction in enumerate(left_predictions):
        print(f"Sentence {i}: {prediction}")

    print("Right model predictions:")
    for i, prediction in enumerate(right_predictions):
        print(f"Sentence {i}: {prediction}")
