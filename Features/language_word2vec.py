# Python codes for extracting language-based word2vec features
from nltk.tokenize import word_tokeniz
from gensim.models import Word2Vec

for session in "raw_text_corpus":
    tokenized_session = []
    for utterance in session:
        if (utterance[0] != utterance[0]): # check if utterance = ['nan']
            utterance = ['empty']
        words = word_tokenize(utterance[0])
        word2vec_model_training_corpus += words
        tokenized_session.append(words)
    text_corpus.append(tokenized_session)

word2vec_model = Word2Vec([word2vec_model_training_corpus], min_count = 1, size = 100, window = 5)
word_embedding = word2vec_model['a_specific_word'] # generate word embedding for each word