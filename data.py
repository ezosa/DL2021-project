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
        dir_path = filepath + dir if filepath.endswith("/") else filepath + "/" + dir
        xml_files = os.listdir(dir_path)
        for f in xml_files:
            art_file = open(dir_path + "/" + f, 'r', encoding='ISO-8859-1')
            print(" [*] Reading articles:", len(articles), end="\r")
            try:
                tree = ET.parse(art_file)
                article_dict = {}
                root = tree.getroot()
                article_dict['id'] = root.attrib['itemid']
                article_dict['date'] = root.attrib['date']
                article_dict['codes'] = []

                for child in root:
                    if "headline" in child.tag:
                        article_dict['headline'] = child.text
                    elif "text" in child.tag:
                        article_text = ''
                        for par in child:
                            article_text += par.text + " "
                        article_dict['text'] = article_text
                    # getting labels
                    elif "metadata" in child.tag:
                        for par in child:
                            if "codes" == par.tag and "topics" in par.attrib["class"]:
                                    article_dict['codes'] = [code.attrib["code"] for code in par]                    
                articles.append(article_dict)

            except Exception as ex:
                print("Error parsing file", dir_path+'/'+f)
                print(ex)
                errors.append(dir_path+'/'+f)
    return articles

if __name__ == "__main__":
    import sys
    import time
    now = time.time()
    data = parse_reuters_articles(sys.argv[1])
    print(f"\n [!] Took {time.time() - now:.2f}s")
    import pdb; pdb.set_trace()

