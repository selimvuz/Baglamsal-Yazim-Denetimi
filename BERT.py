import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Veri kümesini yükle
data = pd.read_csv('dataset/baglamsal.csv')

# Model ve tokenizer'ı yükle
model_name = "ytu-ce-cosmos/turkish-base-bert-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def find_difference(hatali_sentence, dogru_sentence):
    hatali_words = hatali_sentence.split()
    dogru_words = dogru_sentence.split()

    if len(hatali_words) != len(dogru_words):
        print("Hatalı ve doğru cümle farklı uzunlukta.")
        return None, None, None

    for i in range(len(hatali_words)):
        if hatali_words[i] != dogru_words[i]:
            return i, dogru_words[i], hatali_words[i]
    return None, None, None

def mask_and_predict(sentence, mask_index, top_k=5):
    words = sentence.split()
    words[mask_index] = tokenizer.mask_token
    masked_sentence = " ".join(words)

    tokens = tokenizer(masked_sentence, return_tensors='pt')
    mask_token_index = torch.where(tokens.input_ids[0] == tokenizer.mask_token_id)[0]

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    mask_token_logits = logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

    return [tokenizer.decode([token]).strip() for token in top_k_tokens]

def calculate_metrics(data, top_k=5):
    reciprocal_ranks = []
    correct_predictions = 0
    total_predictions = 0
    first_prediction_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for idx, row in data.iterrows():
        mask_index, correct_word, wrong_word = find_difference(row['hatali_cumle'], row['dogru_cumle'])
        if mask_index is not None and correct_word is not None:
            predictions = mask_and_predict(row['hatali_cumle'], mask_index, top_k)
            if correct_word.lower() in predictions:
                rank = predictions.index(correct_word.lower()) + 1
                reciprocal_ranks.append(1 / rank)
                correct_predictions += 1
                if predictions[0].lower() == correct_word.lower():  # İlk tahminin doğruluğunu kontrol etme
                    first_prediction_correct += 1
                true_positive += 1
            else:
                reciprocal_ranks.append(0)
                false_positive += 1
                false_negative += 1
            total_predictions += 1
            # Her veri için çıktıyı yazdır
            print(f"Hatalı Kelime: {wrong_word}, Doğru Kelime: {correct_word}, Modelin Tahminleri: {predictions}, Rank: {1 / rank if correct_word.lower() in predictions else 0}")
        else:
            print(f"Hatalı Kelime: {wrong_word}, Doğru Kelime: {correct_word}, Modelin Tahminleri: None")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    accuracy = first_prediction_correct / total_predictions if total_predictions > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return mrr, accuracy, precision, recall, f1

# MRR, doğruluk, precision, recall ve F1 hesapla
mrr, accuracy, precision, recall, f1 = calculate_metrics(data, top_k=100)

# Pandas ayarlarını değiştir (Konsola verinin tamamını yazsın)
pd.set_option('display.max_colwidth', None)

# MRR, doğruluk, precision, recall ve F1 yazdır
print(f"Genel Doğruluk (MRR): {mrr}")
print(f"Genel Doğruluk (Accuracy): {accuracy}")
print(f"Genel Doğruluk (Precision): {precision}")
print(f"Genel Doğruluk (Recall): {recall}")
print(f"Genel Doğruluk (F1): {f1}")