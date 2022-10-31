from asyncio.windows_events import NULL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xmltodict
import pprint
import re
import time
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import *
import pickle
import json
plt.style.use('ggplot')

class pos_index:
    def __init__(self, df, pos_dic) -> None:
        self.df = df
        self.pos_dic = pos_dic
    
    def plus(self, pos_dic):
        self.df += 1
        (key,value), = pos_dic.items()
        if key in self.pos_dic.keys():
            self.pos_dic[key].append(value)
        else:
            self.pos_dic[key] = [value]

    def load_inf(self):
        return self.df, self.pos_dic

def load_stopwords(path='collections\\stopping_words.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        sw = f.readlines()
    sw_list = []
    for line in sw:
        line = line.replace('\n',"")
        line = line.strip()
        sw_list.append(line)
    return sw_list

def preprocessing(content, stopping='collections\\stopping_words.txt', stemming='porter', tokenization_rule='[^\w\s]'):
    punct = re.compile(tokenization_rule) # regax of token
    content = re.sub(punct, "", content) # tokenisation
    content = content.lower()
    content = content.split() # for stopping and stemming
    if stopping is not None:
        sw = load_stopwords(stopping)
        content = [word for word in content if word not in sw] # sw
    stemmer = SnowballStemmer(stemming)
    content = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in content] # stemming
    content = ' '.join(content)
    return content


def load_xml(path, stoping='collections\\stopping_words.txt', stemming='porter', tokenization_rule='[^\w\s]'):
    with open(path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
        # load total text in the dict_file
    dict_file = xmltodict.parse(xml_file)
    dict_file = dict_file['document']['DOC']
    dic = {}
    for file in dict_file:
        content = file['HEADLINE'] + ' ' + file['TEXT'] # for trec.sample.xml
        # preprocessing for xml file
        new_content = preprocessing(content, stoping, stemming, tokenization_rule)
        dic['{}'.format(file['DOCNO'])] = new_content
    return dic

def find_uniq(dic):
    # find all uniq terms in all documents
    terms = {}
    for docno in dic:
        position = 0
        for term in dic[docno].split():
            position += 1
            if term not in terms:
                terms[term] = pos_index(1,{docno:[position]})
            else:
                terms[term].plus({docno:position})
    return terms
# too slow!!!! O(n^3)
# def create_position_index(terms_dic, dic):
#     # positional index implementation
#     collection_dic = {}
#     for term_hash in terms_dic:
#         doc_dic = {}
#         for docno in dic:
#             # cannot use string.find() --> string index: char, list index: word(str)
#             pos_list = [index for index, word in enumerate(dic[docno].split()) if terms_dic[term_hash] == word]
#             if len(pos_list):
#                 doc_dic[docno] = pos_list
#         collection_dic[terms_dic[term_hash]] = doc_dic
#     return collection_dic

def create_position_index(terms):
    collection_dic = {}
    for term in terms.keys():
        df, pos_dic = terms[term].load_inf()
        collection_dic[term] = [df, pos_dic]
    return collection_dic


def save_position_index(collection_dic, path='collections\\trec_preprocess'):
    # write into txt and binary file
    with open(path + ".txt", 'w') as f1:
        for term in collection_dic.keys():
            f1.write(term+': {}\n'.format(collection_dic[term][0]))
            for docno in collection_dic[term][1].keys():
                f1.write('{}: '.format(docno)+str(collection_dic[term][1][docno])[1:-1]+'\n')
    with open(path + ".dat", 'wb') as f2:
        pickle.dump(collection_dic, f2)

def load_position_index(path='collections\\trec_preprocess'):
    with open('collections/trec.english_stemmer.dat', 'rb') as f:
        data = pickle.load(f)
    return data
