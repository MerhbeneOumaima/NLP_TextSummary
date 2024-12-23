import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from transformers import pipeline

# Fonction de résumé avec spaCy
def summarizer(rawdocs):
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
    summary = ' '.join(final_summary)

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))

# Fonction de résumé avec Hugging Face
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarized = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summarized[0]['summary_text']

# Menu ou choix de méthode de résumé
def main():
    text = input("Entrez le texte à résumer :\n")

    print("\nChoisissez la méthode de résumé :")
    print("1. Résumé basé sur spaCy")
    print("2. Résumé basé sur Hugging Face (facebook/bart-large-cnn)")
    choice = input("Votre choix (1 ou 2) : ")

    if choice == "1":
        summary, doc, original_length, summary_length = summarizer(text)
        print("\n--- Résumé avec spaCy ---")
        print(summary)
        print(f"\nLongueur originale : {original_length} mots")
        print(f"Longueur du résumé : {summary_length} mots")

    elif choice == "2":
        summary = summarize_text(text)
        print("\n--- Résumé avec Hugging Face ---")
        print(summary)

    else:
        print("\nChoix invalide. Veuillez choisir 1 ou 2.")

if __name__ == "__main__":
    main()