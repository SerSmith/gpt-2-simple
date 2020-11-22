from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
import string
import nltk

def remove_multiple_strings(cur_string, replace_list):
    for cur_word in replace_list:
        cur_string = cur_string.replace(cur_word, '')
    return cur_string

def get_text_vectors(data, ignore_words=None, n_features=2**20):
    if ignore_words is None:
        ignore_words = ["<|startoftext|>", "<|endoftext|>"]
    data = [remove_multiple_strings(s, ignore_words) for s in data]
    lemmatizer = nltk.WordNetLemmatizer()
    data_cleaned = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in data]
    data_cleaned = [' '.join([lemmatizer.lemmatize(w) for w in s.split(' ')]) for s in  data_cleaned]
    vectorizer = HashingVectorizer(n_features=n_features,
                                   binary=True,
                                   norm=None,
                                   stop_words=None,
                                   tokenizer=nltk.word_tokenize)
    X = vectorizer.fit_transform(data_cleaned)
    return X


def get_simmilarity(first, second):
    
    word_quant_first = first.sum(axis=1)
    word_quant_second = second.sum(axis=1)

    max_phrase_length = np.zeros((word_quant_first.shape[0], word_quant_second.shape[0]))

    for i in range(word_quant_first.shape[0]):
        for j in range(word_quant_second.shape[0]):
            max_phrase_length[i, j] = max(word_quant_first[i], word_quant_second[j])

    matched_words_quant = (first * second.T) / max_phrase_length

    return matched_words_quant