import spacy
import re

nlp = spacy.load("en_core_web_sm")

class JobParser:

    def __init__(self, raw_text):
        self.text = raw_text

    def clean_text(self):
        text = self.text.lower() 
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9., ]', '', text)

        doc = nlp(text)

        words = []
        for token in doc:
            if not token.is_stop:
                words.append(token.text)

        # Lemmatization
        lemmas = []
        for token in nlp(" ".join(words)):
            lemmas.append(token.lemma_)

        text = " ".join(lemmas)
        return text.strip()