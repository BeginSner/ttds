import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import re
from nltk.stem import *
from gensim.parsing.preprocessing import remove_stopwords

plt.style.use('ggplot')
def search(input, data):
    input = input.lower()
    stemmer = SnowballStemmer("english")
    punct = re.compile("[^\w\s]") # regax of token
    reg_quo = re.compile(r'"(.*?)"') # in ""
    reg_bra = re.compile(r"^#\d{1,}\((.*?)[)]") # begin with # and at least 1 number and in ()
    # reg_num = re.compile(r"\d{1,}") # distance
    if ' and not ' in input:
        if reg_quo.findall(input):
            return boolearn_search(input, data, reg_quo, label='and not')
        elif reg_bra.findall(input):
            return boolearn_search(input, data, reg_bra, label='and not')
        else:
            return boolearn_search(input, data, label='and not')
    elif ' and ' in input:
        if reg_quo.findall(input):
            return boolearn_search(input, data, reg_quo, label='and')
        elif reg_bra.findall(input):
            return boolearn_search(input, data, reg_bra, label='and')
        else:
            return boolearn_search(input, data, label='and')
    if ' or ' in input:
        if reg_quo.findall(input):
            return boolearn_search(input, data, reg_quo, label='or')
        elif reg_bra.findall(input):
            return boolearn_search(input, data, reg_bra, label='or')
        else:
            return boolearn_search(input, data, label='or')
    else:
        if reg_quo.findall(input):
            return boolearn_search(input, data, reg_quo)
        elif reg_bra.findall(input):
            return boolearn_search(input, data, reg_bra)
        else:
            return boolearn_search(input, data)
        
def boolearn_search(input, data, reg=None, label:'str'=None):
    input = input.lower()
    stemmer = SnowballStemmer("english")
    reg_quo = re.compile(r'"(.*?)"') # in ""
    reg_bra = re.compile(r"^#\d{1,}\((.*?)[)]") # begin with # and at least 1 number and in ()
    punct = re.compile("[^\w\s]") # regax of token
    if reg == None:
        reg = reg_quo

    if label == None:
        if not reg.findall(input):
            sentence = re.sub(punct, "", input) # remove ,
            sentence = remove_stopwords(sentence)
            sentence = sentence.split()
            sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence]
            public_key = data[sentence[0]].keys()
            for term in sentence:
                public_key = data[term].keys() & public_key
            sentence = ' '.join(sentence)
            return public_key, sentence
        elif reg.findall(input):
            if reg == reg_quo:
                return phrase_search(input, data)
            else: 
                return proximity_search(input, data)
            
        
    elif label == 'and':
        split_list = input.split(" and ")
    elif label == 'or':
        split_list = input.split(" or ")
    elif label == 'and not':
        split_list = input.split(" and not ")
    else: 
        print("error label!")

    
    
    
    if not reg.findall(split_list[0]) and not reg.findall(split_list[1]):
        sentence1 = re.sub(punct, "", split_list[0]) # remove ,
        sentence1 = remove_stopwords(sentence1)
        sentence1 = sentence1.split()
        sentence1 = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence1]
        public_key1 = data[sentence1[0]].keys()
        for term in sentence1:
            public_key1 = data[term].keys() & public_key1
        sentence1 = ' '.join(sentence1)

        sentence2 = re.sub(punct, "", split_list[1]) # remove ,
        sentence2 = remove_stopwords(sentence2)
        sentence2 = sentence2.split()
        sentence2 = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence2]
        public_key2 = data[sentence2[0]].keys()
        for term in sentence2:
            public_key2 = data[term].keys() & public_key2
        sentence2 = ' '.join(sentence2)
        
        if label == 'and not':
            public_key = public_key1 - public_key2
            sentence = sentence1
        elif label == 'and':
            public_key = public_key1 & public_key2
            sentence = sentence1 + ' ' + sentence2
        elif label == 'or':
            public_key = public_key1 | public_key2
            sentence = sentence1 + ' ' + sentence2
        return public_key, sentence1

    elif reg.findall(split_list[0]) and reg.findall(split_list[1]):
        if reg == reg_quo:
            doc_dic1, sentence1 = phrase_search(split_list[0], data)
            doc_dic2, sentence2 = phrase_search(split_list[1], data)
        else: 
            doc_dic1, sentence1 = proximity_search(split_list[0], data)
            doc_dic2, sentence2 = proximity_search(split_list[1], data)
        if label == 'and not':
            public_key = doc_dic1.keys() - doc_dic2.keys()
            sentence = sentence1
        elif label == 'and':
            public_key = doc_dic1.keys() & doc_dic2.keys()
            sentence = sentence1 + ' ' + sentence2
        elif label == 'or':
            public_key = doc_dic1.keys() | doc_dic2.keys()
            sentence = sentence1 + ' ' + sentence2
        return public_key, sentence

    elif not reg.findall(split_list[0]) and reg.findall(split_list[1]):
        sentence = re.sub(punct, "", split_list[0]) # remove ,
        sentence = remove_stopwords(sentence)
        sentence = sentence.split()
        sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence]
        public_key = data[sentence[0]].keys()
        for term in sentence:
            public_key = data[term].keys() & public_key
        sentence = ' '.join(sentence)

        if reg == reg_quo:
            doc_dic2, sentence2 = phrase_search(split_list[1], data)
        else:
            doc_dic2, sentence2 = proximity_search(split_list[1], data)
        if label == 'and not':
            public_key = public_key1 - doc_dic2.keys()
            sentence = sentence
        elif label == 'and':
            public_key = public_key1 & doc_dic2.keys()
            sentence = sentence + ' ' + sentence2
        elif label == 'or':
            public_key = public_key1 | doc_dic2.keys()
            sentence = sentence + ' ' + sentence2
        return public_key, sentence

    elif reg.findall(split_list[0]) and not reg.findall(split_list[1]):
        if reg == reg_quo:
            doc_dic1, sentence1 = phrase_search(split_list[0], data)
        else: 
            doc_dic1, sentence1 = proximity_search(split_list[0], data)
        sentence = re.sub(punct, "", split_list[1]) # remove ,
        sentence = remove_stopwords(sentence)
        sentence = sentence.split()
        sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence]
        public_key = data[sentence[0]].keys()
        for term in sentence:
            public_key = data[term].keys() & public_key
        sentence = ' '.join(sentence)

        
        if label == 'and not':
            public_key = doc_dic1.keys() - public_key
            sentence = sentence1
        elif label == 'and':
            public_key = doc_dic1.keys() & public_key
            sentence = sentence1 + ' ' + sentence
        elif label == 'or':
            public_key = doc_dic1.keys() | public_key
            sentence = sentence1 + ' ' + sentence
        return public_key, sentence
        
    else: 
        print("error")
    
def phrase_search(input, data):
    doc_dic = {}
    reg_quo = re.compile(r'"(.*?)"') # in ""
    input = input.lower()
    sentence = reg_quo.findall(input)[0]
    distance = 5
    stemmer = SnowballStemmer("english")
    punct = re.compile("[^\w\s]") # regax of token
    sentence = re.sub(punct, "", sentence) # remove ,
    sentence = remove_stopwords(sentence)
    sentence = sentence.split()
    sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence]
    public_key = data[sentence[0]].keys() & data[sentence[1]].keys() # find common keys for two words
    for key in public_key:
        for index0 in data[sentence[0]][key]:
            for index1 in data[sentence[1]][key]:
                if np.abs(index0-index1) < distance:
                    doc_dic[key] = [index0, index1]
    sentence = ' '.join(sentence)
    return doc_dic, sentence

def proximity_search(input, data):
    doc_dic = {}
    reg_bra = re.compile(r"^#\d{1,}\((.*?)[)]") # begin with # and at least 1 number and in ()
    reg_num = re.compile(r"\d{1,}") # distance
    input = input.lower()
    sentence = reg_bra.findall(input)[0]
    distance = int(reg_num.findall(input)[0])
    stemmer = SnowballStemmer("english")
    punct = re.compile("[^\w\s]") # regax of token
    sentence = re.sub(punct, "", sentence) # remove ,
    sentence = remove_stopwords(sentence)
    sentence = sentence.split()
    sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence]
    public_key = data[sentence[0]].keys() & data[sentence[1]].keys() # find common keys for two words
    for key in public_key:
        for index0 in data[sentence[0]][key]:
            for index1 in data[sentence[1]][key]:
                if np.abs(index0-index1) < distance:
                    doc_dic[key] = [index0, index1]
    sentence = ' '.join(sentence)
    return doc_dic, sentence

