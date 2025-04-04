from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Kaydedilen modelin ve sözlüklerin yüklenmesi
model = load_model("pos_tag.h5")

# Kelime ve etiket sözlüklerinin yüklenmesi
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("tag2idx.pkl", "rb") as f:
    tag2idx = pickle.load(f)

# Kelime ve etiket dizilerini oluşturma
idx2tag = {i: t for t, i in tag2idx.items()}

def predict_sentence(sentence, model, word2idx, idx2tag, max_len):
    '''
        Modelin Predict fonsksiyonu ile verilen cümledeki kelimelerin POS etiketlerini tahmin eder.
        sentence: str, tahmin edilecek cümle
        model: Keras model nesnesi, tahmin için kullanılacak model
        word2idx: dict, kelime indeks sözlüğü
        idx2tag: dict, etiket indeks sözlüğü
        max_len: int, modelin eğitiminde kullanılan maksimum kelime sayısı
        return: list of tuples, (kelime, tahmin edilen etiket) çiftleri
    '''

    tokens = sentence.strip().split()

    seq = [word2idx.get(token, word2idx["UNK"]) for token in tokens]
    seq_padded = pad_sequences([seq], maxlen=max_len, padding="post", value=word2idx["PAD"])

    pred = model.predict(seq_padded)
    pred_idx = np.argmax(pred, axis=-1)[0]
    pred_tags = [idx2tag.get(idx, "UNK") for idx in pred_idx][:len(tokens)]

    return list(zip(tokens, pred_tags))

# modelin içerisindeki tüm tagları ekrana verir (isteğe bağlı)
for tag, idx in tag2idx.items():
    print(f"{tag}: {idx}")


while True:
    example_sentence = input("Cümleyi girin (çıkmak için 'exit' yazın): ")
    if example_sentence.lower() == "exit":
        break

    predictions = predict_sentence(example_sentence, model, word2idx, idx2tag, max_len=50)

    print("Cümle ve tahmin edilen POS etiketleri:")
    
    for word, pos in predictions:
        print(f"{word}\t{pos}")
