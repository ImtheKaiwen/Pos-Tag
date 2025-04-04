import numpy as np
from conllu import parse
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def processData(file_path):
    '''
        Veriyi okuyup, cümleleri ve etiketleri ayırır.
        file_path: str, veri dosyasının yolu
        return: tuple, (kelimeler, etiketler)
    '''
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    sentences = parse(data)

    word_sentences = []
    pos_sentences = []

    for sent in sentences:
        words = []
        pos_tags = []
        for token in sent:
            words.append(token["form"])
            pos_tags.append(token.get("upos", "UNK"))
        word_sentences.append(words)
        pos_sentences.append(pos_tags)

    words = {w for sent in word_sentences for w in sent}
    tags = {t for sent in pos_sentences for t in sent}

    word2idx = {w: i + 2 for i, w in enumerate(sorted(words))}
    word2idx["PAD"] = 0
    word2idx["UNK"] = 1

    tag2idx = {t: i + 1 for i, t in enumerate(sorted(tags))}
    tag2idx["PAD"] = 0

    X = [[word2idx.get(w, word2idx["UNK"]) for w in sent] for sent in word_sentences]
    y = [[tag2idx[t] for t in sent] for sent in pos_sentences]

    max_len = max(len(s) for s in X)
    X = pad_sequences(X, maxlen=max_len, padding="post", value=word2idx["PAD"])
    y = pad_sequences(y, maxlen=max_len, padding="post", value=tag2idx["PAD"])

    num_tags = len(tag2idx)
    y = [to_categorical(seq, num_classes=num_tags) for seq in y]
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, word2idx, tag2idx, max_len, num_tags

def Train(X_train, y_train, X_val, y_val, word2idx, tag2idx, max_len, num_tags):
    '''
        Modeli tanımlar ve eğitir.
        X_train: Eğitim verisi (kelimeler)
        y_train: Eğitim etiketleri (POS etiketleri)
        X_val: Doğrulama verisi (kelimeler)
        y_val: Doğrulama etiketleri (POS etiketleri)
        word2idx: Kelime indeks sözlüğü
        tag2idx: Etiket indeks sözlüğü
        max_len: Maksimum cümle uzunluğu
        num_tags: Etiket sayısı
        return: Eğitim geçmişi
    '''
    model = Sequential()
    model.add(Embedding(input_dim=len(word2idx), output_dim=128, mask_zero=False))  
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1)))  
    model.add(TimeDistributed(Dense(num_tags, activation="softmax")))

    model.compile(optimizer="adam", 
                loss="categorical_crossentropy", 
                metrics=["accuracy"],
                run_eagerly=True)

    model.summary()

    history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_val, y_val))

    model.save("pos_tag.h5")
    print("Model kaydedildi: pos_tag.h5")

    with open("word2idx.pkl", "wb") as f:
        pickle.dump(word2idx, f)
    with open("tag2idx.pkl", "wb") as f:
        pickle.dump(tag2idx, f)

    idx2tag = {i: t for t, i in tag2idx.items()}

    return model, history, word2idx, tag2idx, idx2tag

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

def visualize(history):
    '''
        Eğitim sürecini görselleştirir.
        history: Keras model eğitim geçmişi
    '''
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], marker='o', label='Eğitim Accuracy')
    plt.plot(history.history['val_accuracy'], marker='x', label='Doğrulama Accuracy')
    plt.title("Eğitim ve Doğrulama Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], marker='o', label='Eğitim Loss')
    plt.plot(history.history['val_loss'], marker='x', label='Doğrulama Loss')
    plt.title("Eğitim ve Doğrulama Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


Xtrain, X_val, y_train, y_val, word2idx, tag2idx, max_len, num_tags = processData("tr_boun-ud-train.conllu")
model, history, word2idx, tag2idx, idx2tag = Train(Xtrain, y_train, X_val, y_val, word2idx, tag2idx, max_len, num_tags)
visualize(history)

example_sentence = "Yeni emekli olmuş general bir dostum, taşındığı evinde plaketleri asacak."
predictions = predict_sentence(example_sentence, model, word2idx, idx2tag, max_len)

print("Cümle ve tahmin edilen POS etiketleri:")
for word, pos in predictions:
    print(f"{word}\t{pos}")


