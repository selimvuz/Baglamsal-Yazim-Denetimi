# Bağlam Tabanlı Yazım Denetiminde BERT ve GPT Modellerinin Karşılaştırması

Bu projede, kelimeleri yanlış ve doğru bağlamlarda kullanarak bir veri kümesi hazırlayıp bu veriler üzerinde BERT ve GPT modellerinin hatalı kelimeleri düzeltip doğru cümlelerdeki hallerine ne kadar iyi getirebildiklerinin farklı metrikler üzerinden karşılaştırmasını yapacağız.

### Gerekli Kütüphaneler

Python kodlarını çalıştırmak için öncelikle gerekli kütüphaneleri yüklemelisiniz:

```bash
pip install pandas transformers torch
```

### Veri Kümesi

Türkçe'de bağlamsal yazım denetimi için yeterince açık kaynak bulunmadığından kendimiz hazırlamak istedik. Örneklere **dataset** klasöründen ulaşabilirsiniz.

| Veriler |                    Hatalı Cümle                    |                    Doğru Cümle                     |
| :-----: | :------------------------------------------------: | :------------------------------------------------: |
|   #1    |   Ondan bir şey istediğime çok pişmaniye oldum.    |     Ondan bir şey istediğime çok pişman oldum.     |
|   #2    |         Kaza sonucunda fren raydan çıktı.          |         Kaza sonucunda tren raydan çıktı.          |
|   #3    |    Bazen emeklemeden coşmayı öğrenmen gerekir.     |    Bazen emeklemeden koşmayı öğrenmen gerekir.     |
|   #4    |     Hatay usulü hepsi kebabı tarifi arıyorum.      |     Hatay usulü tepsi kebabı tarifi arıyorum.      |
|   #5    | Garaj kapısının önüne çark edenler cezalandırılır. | Garaj kapısının önüne park edenler cezalandırılır. |
