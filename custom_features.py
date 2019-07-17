import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import lexicon_reader


class CustomFeats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def __init__(self, filedir):
        self.feat_names = set()
        lexicon_dir = os.path.join(filedir, 'lexicon')
        self.inqtabs_dict = lexicon_reader.read_inqtabs(os.path.join(lexicon_dir, 'inqtabs.txt'))
        self.swn_dict = lexicon_reader.read_senti_word_net(os.path.join(lexicon_dir, 'SentiWordNet_3.0.0_20130122.txt'))

    def fit(self, x, y=None):
        return self

    @staticmethod
    def word_count(review):
        words = review.split(' ')
        return len(words)

    def pos_count(self, review):
        words = review.split(' ')
        count = 0
        for word in words:
            if word in self.inqtabs_dict.keys() and self.inqtabs_dict[word] == lexicon_reader.POS_LABEL:
                count += 1
        return count

    def features(self, review):
        return {
            'length': len(review),
            'num_sentences': review.count('.'),
            'num_words': self.word_count(review),
            'pos_count': self.pos_count(review)  # 4 example features; add your own here e.g. word_count, and pos_count
        }

    def get_feature_names(self):
        return list(self.feat_names)

    def transform(self, reviews):
        feats = []
        for review in reviews:
            f = self.features(review)
            [self.feat_names.add(k) for k in f]
            feats.append(f)
        return feats


def get_custom_vectorizer():
    # Experiment with different vectorizers
    #    # Experiment with different vectorizers
     #unigram = CountVectorizer(ngram_range=(1,1))
     #bigram = CountVectorizer(ngram_range=(2,2))
    #trigram = CountVectorizer(ngram_range=(3,3))
#    #fourgram = CountVectorizer(ngram_range=(4,4))
#    combined = CountVectorizer(ngram_range=(1,3))    
    tfidf = TfidfVectorizer()
    return CountVectorizer()
