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


def create_vocab_from_data(ft_file,  min_df=5, max_df=0.8):
    df = pd.read_feather(ft_file)
    texts = list(df.text)
    #headlines = list(df.headline)

    clean_docs = [clean_document(doc) for doc in texts]

    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
    cvectorizer.fit_transform(clean_docs).sign()

    word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
    id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
    print("Vocab size:", len(word2id))

    vocab_file = ft_file[:-3] + "_vocab.pkl"
    with open(vocab_file, 'wb') as pf:
        pickle.dump(word2id, pf)
        print("Done! Saved vocab as", vocab_file)


