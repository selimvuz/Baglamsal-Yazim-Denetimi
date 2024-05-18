import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Veri kümesini yükle
data = pd.read_csv('dataset/baglamsal.csv')

# Model ve tokenizerı yükle
model_name = "ytu-ce-cosmos/turkish-base-bert-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def find_difference(hatali_sentence, dogru_sentence):
    hatali_words = hatali_sentence.split()
    dogru_words = dogru_sentence.split()

    if len(hatali_words) != len(dogru_words):
        print("Hatalı ve doğru cümle farklı uzunlukta.")
        return None, None

    for i in range(len(hatali_words)):
        if hatali_words[i] != dogru_words[i]:
            return i, dogru_words[i], hatali_words[i]
    return None, None, None

def mask_and_predict(sentence, mask_index, top_k=5):
    words = sentence.split()
    words[mask_index] = tokenizer.mask_token
    masked_sentence = " ".join(words)

    tokens = tokenizer(masked_sentence, return_tensors='pt')
    mask_token_index = torch.where(
        tokens.input_ids[0] == tokenizer.mask_token_id)[0]

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    mask_token_logits = logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, top_k,
                              dim=1).indices[0].tolist()

    return [tokenizer.decode([token]).strip() for token in top_k_tokens]

def calculate_mrr(data, top_k):
    reciprocal_ranks = []

    for idx, row in data.iterrows():
        mask_index, correct_word, wrong_word = find_difference(
            row['hatali_cumle'], row['dogru_cumle'])
        if mask_index is not None and correct_word is not None:
            predictions = mask_and_predict(
                row['hatali_cumle'], mask_index, top_k)
            if correct_word in predictions:
                rank = predictions.index(correct_word) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)
            # Her veri için çıktıyı yazdır
            print(
                f"Hatalı Kelime: {wrong_word}, Doğru Kelime: {correct_word}, Modelin Tahminleri: {predictions}")
        else:
            print(
                f"Hatalı Kelime: {wrong_word}, Doğru Kelime: {correct_word}, Modelin Tahminleri: None")

    mrr = sum(reciprocal_ranks) / \
        len(reciprocal_ranks) if reciprocal_ranks else 0
    return mrr

# MRR hesapla
mrr = calculate_mrr(data, top_k=5)

# Pandas ayarlarını değiştir (Konsola verinin tamamını yazsın)
pd.set_option('display.max_colwidth', None)

# MRR yazdır
print(f"Genel Doğruluk (MRR): {mrr}")
