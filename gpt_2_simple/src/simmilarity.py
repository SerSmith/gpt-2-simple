from sklearn.feature_extraction.text import HashingVectorizer
from nltk.tokenize import NLTKWordTokenizer
import numpy as np
import string
from nltk import WordNetLemmatizer

def get_text_vectors(data, n_features=2**20):
    lemmatizer = WordNetLemmatizer()
    data_cleaned = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in data]
    data_cleaned = [' '.join([lemmatizer.lemmatize(w) for w in s.split(' ')]) for s in  data_cleaned]
    print(data_cleaned)
    tk = NLTKWordTokenizer()
    vectorizer = HashingVectorizer(n_features=n_features,
                                   binary=True,
                                   norm=None,
                                   stop_words=None,
                                   tokenizer=tk.tokenize)
    X = vectorizer.fit_transform(data_cleaned)
    return X


def get_simmilartity(first, second):
    
    word_quant_first = first.sum(axis=1)
    word_quant_second = second.sum(axis=1)

    max_phrase_length = np.zeros((word_quant_first.shape[0], word_quant_second.shape[0]))

    for i in range(word_quant_first.shape[0]):
        for j in range(word_quant_second.shape[0]):
            max_phrase_length[i, j] = max(word_quant_first[i], word_quant_second[j])

    matched_words_quant = (first * second.T) / max_phrase_length

    return matched_words_quant

X1 = get_text_vectors(['I am groot qwerty', 'Am I crazy?', "One Two Three"] * 4000)
X2 = get_text_vectors(['I am groot', 'Am I crazy?', "One Two Three"])
print(get_simmilartity(X1, X2))
# print(X * X.T)
# print(X.sum(axis=1))