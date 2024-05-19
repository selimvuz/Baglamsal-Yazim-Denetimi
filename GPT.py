import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Veri kümesini yükle
data = pd.read_csv('dataset/baglamsal.csv')

# Model ve tokenizer'ı yükle
model_name = "ytu-ce-cosmos/turkish-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Pad token ekle
tokenizer.pad_token = tokenizer.eos_token

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
    masked_sentence = " ".join(words[:mask_index])
    if masked_sentence.strip() == "":  # Boş girişleri kontrol et
        return []

    inputs = tokenizer.encode(masked_sentence, return_tensors='pt')
    
    if inputs.size(1) == 0:  # Giriş boyutunu kontrol et
        return []

    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs.logits

    predicted_indices = torch.topk(predictions[0, -1, :], top_k).indices.tolist()
    predicted_tokens = [tokenizer.decode([idx]).strip() for idx in predicted_indices]

    return predicted_tokens

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
            if predictions:
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
mrr, accuracy, precision, recall, f1 = calculate_metrics(data, top_k=5)

# Pandas ayarlarını değiştir (Konsola verinin tamamını yazsın)
pd.set_option('display.max_colwidth', None)

# MRR, doğruluk, precision, recall ve F1 yazdır (Yüzdesel olarak)
print(f"Genel Doğruluk (MRR): {mrr}")
print(f"Genel Doğruluk (Accuracy): {accuracy * 100:.2f}%")
print(f"Genel Doğruluk (Precision): {precision * 100:.2f}%")