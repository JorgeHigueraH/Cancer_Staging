import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TumorRowlandPreprocessor:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.medical_noise = {'received', 'fresh', 'frozen', 'section', 'patient', 'date', 'page', 'copy', 'material', 'examination'}
        self.stop_words.update(self.medical_noise)
        self.lemmatizer = WordNetLemmatizer()

    def clean_for_baseline(self, text):
        text = str(text).lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r'\d+', '[NUM]', text)
        tokens = [self.lemmatizer.lemmatize(t) for t in text.split() if t not in self.stop_words]
        return " ".join(tokens)

    def clean_for_trees(self, text):
        text = str(text).lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        tokens = [self.lemmatizer.lemmatize(t) for t in text.split() if t not in self.medical_noise]
        return " ".join(tokens)

    def clean_for_sequence_models(self, text):
        text = str(text).lower()
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        tokens = [t for t in text.split() if t not in self.medical_noise]
        return " ".join(tokens)

    def clean_for_transformers(self, text):
        text = str(text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'(page \d+ / \d+|copy no\.? \d+)', '', text, flags=re.IGNORECASE)
        return text.strip()