from flask import Flask, render_template, request
from transformers import pipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from PyPDF2 import PdfReader

app = Flask(__name__)

# Chargement du modèle Hugging Face pour le résumé
huggingface_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Fonction de résumé avec spaCy
def spacy_summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(rawdocs)
    word_freq = {}

    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    sent_tokens = [sent for sent in doc.sents]
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    select_len = int(len(sent_tokens) * 0.3)
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    final_summary = [word.text for word in summary]
    return ' '.join(final_summary)

# Fonction de résumé avec Hugging Face
def huggingface_summary(text):
    return huggingface_summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    rawtext = request.form['rawtext']
    summary = huggingface_summary(rawtext)  # Résumer avec Hugging Face
    return render_template('summary.html', summary=summary, original_txt=rawtext, len_summary=len(summary.split(' ')), len_orig_txt=len(rawtext.split(' ')))

@app.route('/analyze-pdf', methods=['POST'])
def analyze_pdf():
    if 'pdf' not in request.files:
        return "No file uploaded."
    file = request.files['pdf']
    rawtext = extract_text_from_pdf(file)  # Extraire le texte du PDF
    summary = huggingface_summary(rawtext)  # Résumer avec Hugging Face
    return render_template('summary.html', summary=summary, original_txt=rawtext, len_summary=len(summary.split(' ')), len_orig_txt=len(rawtext.split(' ')))

if __name__ == "__main__":
    app.run(debug=True)
