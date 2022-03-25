# Python codes for extracting language-based tf-idf for unigrams and bi-grams

import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
porter = PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

clean_corpus = []

for turn_exchange in turn_pairs: # turn_pairs: corpus for turn exchanges

    turn_exchange = re.sub(r'[,]', '', turn_exchange)
    turn_exchange = re.sub(r'[.]', '', turn_exchange)
    turn_exchange = re.sub(r'[?]', '', turn_exchange)

    turn_exchange = stemSentence(turn_exchange) # stem the sentence for tf-idf
    clean_corpus.append(turn_exchange)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_corpus)
