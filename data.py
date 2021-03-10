from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import os
import string
import re
import json
import random
import pickle
import xml.etree.ElementTree as ET

exclude = set(string.punctuation)
stopwords_en = set(stopwords.words('english'))


def clean_document(doc):
    clean_punc = ''.join(ch if ch not in exclude else '' for ch in doc.lower())
    clean_punc_tokens = word_tokenize(clean_punc)
    clean_stop = [tok for tok in clean_punc_tokens if tok not in stopwords_en and len(tok) > 2][1:]
    #clean_digits = [tok for tok in clean_stop if re.match(r'^([\s\d]+)$', tok) is None]
    return clean_stop


def parse_reuters_articles(filepath):
    direc = os.listdir(filepath)
    direc = [dir for dir in direc if re.match(r'^([\s\d]+)$', dir) is not None]
    articles = []
    errors = []
    for dir in direc:
        dir_path = filepath + dir
        xml_files = os.listdir(dir_path)
        for f in xml_files:
            art_file = open(dir_path + "/" + f, 'r', encoding='utf-8')
            print("Articles:", len(articles))
            try:
                tree = ET.parse(art_file)
                article_dict = {}
                root = tree.getroot()
                article_dict['id'] = root.attrib['itemid']
                article_dict['date'] = root.attrib['date']
                for child in root:
                    if "headline" in child.tag:
                        headline = child
                        article_dict['headline'] = headline.text
                    elif "text" in child.tag:
                        text = child
                        article_text = ''
                        for par in text:
                            article_text += par.text + " "
                        article_dict['text'] = article_text
                articles.append(article_dict)
            except:
                print("Error parsing file", dir_path+'/'+f)
                errors.append(dir_path+'/'+f)
    return articles


# def get_reuters_articles(filepath, max_art=50000):
#     data = pickle.load(open(filepath,"rb"))[:max_art]
#     if "sampled" in filepath:
#         articles = [clean_document(art['text']) for art in data]
#     else:
#         articles = [clean_document(art) for art in data]
#     return articles
