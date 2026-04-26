import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

def _pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    padded_seqs = []
    for seq in sequences:
        if truncating == 'post':
            trunc_seq = seq[:maxlen]
        else:
            trunc_seq = seq[-maxlen:]
        
        if padding == 'post':
            pad_seq = trunc_seq + [value] * max(0, maxlen - len(trunc_seq))
        else:
            pad_seq = [value] * max(0, maxlen - len(trunc_seq)) + trunc_seq
        padded_seqs.append(pad_seq)
    return np.array(padded_seqs)

class PyTorchTokenizer:
    def __init__(self, num_words, oov_token="[UNK]"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.word_counts = Counter()

    def fit_on_texts(self, texts):
        for text in texts:
            self.word_counts.update(str(text).split())
        
        self.word_index = {self.oov_token: 1}
        
        sorted_words = [w for w, _ in self.word_counts.most_common()]
        
        idx = 2
        for word in sorted_words:
            if self.num_words is not None and idx >= self.num_words:
                break
            self.word_index[word] = idx

    def texts_to_sequences(self, texts):
        sequences = []
        oov_idx = self.word_index[self.oov_token]
        for text in texts:
            seq = [self.word_index.get(word, oov_idx) for word in str(text).split()]
            sequences.append(seq)
        return sequences

class TumorRowlandVectorizer:
    def __init__(self, max_features=5000, max_len=500):
        self.max_features = max_features
        self.max_len = max_len
        
        # Vectores para Machine Learning Clásico
        self.tfidf_baseline = TfidfVectorizer(max_features=max_features)
        self.tfidf_trees = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        
        self.tokenizer_seq = PyTorchTokenizer(num_words=max_features, oov_token="[UNK]")

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
        X_train = _pad_sequences(secuencias_train, maxlen=self.max_len, padding='post', truncating='post')
        
        if text_series_test is not None:
            secuencias_test = self.tokenizer_seq.texts_to_sequences(text_series_test)
            X_test = _pad_sequences(secuencias_test, maxlen=self.max_len, padding='post', truncating='post')
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