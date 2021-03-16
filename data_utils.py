import numpy as np
import pandas as pd
import string
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

exclude = set(string.punctuation)
stops = set(stopwords.words('english'))


def clean_document(doc):
    doc = str(doc)
    clean_punc = ''.join(ch if ch not in exclude else '' for ch in doc.lower())
    clean_punc_tokens = clean_punc.split()
    clean_stop = " ".join([tok for tok in clean_punc_tokens if len(tok) > 2 and tok not in stops])
    #clean_digits = " ".join([tok for tok in clean_stop if re.match(r'^([\s\d]+)$', tok) is None])
    return clean_stop


def create_vocab_from_data(csv_file, min_df=5, max_df=0.7):
    print("Loading df:", csv_file)
    df = pd.read_csv(csv_file)
    print("df:", df.shape)
    texts = list(df.text)
    #headlines = list(df.headline)
    clean_docs = [clean_document(doc) for doc in texts]
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
    cvectorizer.fit_transform(clean_docs).sign()
    word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
    print("vocab before OOV:", len(word2id))
    word2id['OOV'] = len(word2id)
    print("vocab + OOV:", len(word2id))
    vocab_file = "vocabulary.pkl"
    with open(vocab_file, 'wb') as pf:
        pickle.dump(word2id, pf)
        print("Done! Saved vocab as", vocab_file)


def create_labels_dictionary(csv_file):
    print("Loading df:", csv_file)
    df = pd.read_csv(csv_file)
    labels = list(df.codes)
    labels_dict = dict([(labels[i], i) for i in range(len(labels))])

    labels_file = "labels_dictionary.pkl"
    with open(labels_file, 'wb') as pf:
        pickle.dump(labels_dict, pf)
        print("Done! Saved labels dictionary as", labels_file)

create_vocab_from_data("data.csv")