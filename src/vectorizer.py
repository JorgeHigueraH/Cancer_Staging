import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TumorRowlandVectorizer:
    def __init__(self, max_features=5000, max_len=500):
        self.max_features = max_features
        self.max_len = max_len
        
        # Vectores para Machine Learning Clásico
        self.tfidf_baseline = TfidfVectorizer(max_features=max_features)
        self.tfidf_trees = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        
        self.tokenizer_seq = Tokenizer(num_words=max_features, oov_token="[UNK]")

    def vectorize_for_baseline(self, text_series_train, text_series_test=None):
        X_train = self.tfidf_baseline.fit_transform(text_series_train)
        if text_series_test is not None:
            X_test = self.tfidf_baseline.transform(text_series_test)
            return X_train, X_test
        return X_train

    def vectorize_for_trees(self, text_series_train, text_series_test=None):
        X_train = self.tfidf_trees.fit_transform(text_series_train)
        if text_series_test is not None:
            X_test = self.tfidf_trees.transform(text_series_test)
            return X_train, X_test
        return X_train

    def vectorize_for_sequence_models(self, text_series_train, text_series_test=None):
        self.tokenizer_seq.fit_on_texts(text_series_train)
        
        secuencias_train = self.tokenizer_seq.texts_to_sequences(text_series_train)
        
        X_train = pad_sequences(secuencias_train, maxlen=self.max_len, padding='post', truncating='post')
        
        if text_series_test is not None:
            secuencias_test = self.tokenizer_seq.texts_to_sequences(text_series_test)
            X_test = pad_sequences(secuencias_test, maxlen=self.max_len, padding='post', truncating='post')
            return X_train, X_test
            
        return X_train

    def get_bioword2vec_matrix(self, word_vectors):
        embedding_matrix = np.zeros((self.max_features, 200))
        
        for word, i in self.tokenizer_seq.word_index.items():
            if i >= self.max_features: 
                continue
            if word in word_vectors:
                embedding_matrix[i] = word_vectors[word]
                
        return embedding_matrix