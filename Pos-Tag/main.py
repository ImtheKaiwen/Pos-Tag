from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import stanza

app = Flask(__name__)
CORS(app)  # Tarayıcıdan gelen istekleri kabul etmek için

# Türkçe NLP modelini yükle
nlp = stanza.Pipeline('tr')

@app.route('/')
def index():
    return render_template('site.html')  # site.html dosyasını döndür

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    sentence = data.get("sentence", "")

    if not sentence:
        return jsonify({"error": "Cümle boş olamaz!"}), 400

    turkce_turler = {
        'NOUN': 'İsim',
        'VERB': 'Fiil',
        'ADJ': 'Sıfat',
        'ADV': 'Zarf',
        'PRON': 'Zamir',
        'DET': 'Belirteç',
        'ADP': 'İlgeç',
        'CONJ': 'Bağlaç',
        'NUM': 'Sayı',
        'PART': 'Parça',
        'INTJ': 'Ünlem',
        'X': 'Bilinmeyen',
        'PUNCT': 'Noktalama'
    }

    doc = nlp(sentence)
    results = [{"word": word.text, "role": turkce_turler.get(word.upos, "Bilinmeyen")}
               for sent in doc.sentences for word in sent.words]

    print(results)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
