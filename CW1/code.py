from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import xmltodict
import re
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import *
import pickle
import json

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

def load_stopwords(path='C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\stopping_words.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        sw = f.readlines()
    sw_list = []
    for line in sw:
        line = line.replace('\n',"")
        line = line.strip()
        sw_list.append(line)
    return sw_list

def preprocessing(content, stopping='C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\stopping_words.txt', stemming='porter', tokenization_rule='[^\w\s]'):
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


def load_xml(path, stoping='C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\stopping_words.txt', stemming='porter', tokenization_rule='[^\w\s]'):
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


def save_position_index(collection_dic, path='C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\trec_preprocess'):
    # write into txt and binary file
    with open(path + ".txt", 'w') as f1:
        for term in collection_dic.keys():
            f1.write(term+': {}\n'.format(collection_dic[term][0]))
            for docno in collection_dic[term][1].keys():
                f1.write('{}: '.format(docno)+str(collection_dic[term][1][docno])[1:-1]+'\n')
    with open(path + ".dat", 'wb') as f2:
        pickle.dump(collection_dic, f2)

def load_position_index(path='C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\trec_preprocess'):
    with open('collections/trec.english_stemmer.dat', 'rb') as f:
        data = pickle.load(f)
    return data

def search(input, data):
    input = input.lower()
    stemmer = SnowballStemmer("porter")
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
    stemmer = SnowballStemmer("porter")
    reg_quo = re.compile(r'"(.*?)"') # in ""
    reg_bra = re.compile(r"^#\d{1,}\((.*?)[)]") # begin with # and at least 1 number and in ()
    punct = re.compile("[^\w\s]") # regax of token
    if reg == None:
        reg = reg_quo

    if label == None:
        if not reg.findall(input):
            sentence = re.sub(punct, " ", input) # remove ,
            sentence = ' '.join(sentence.split())
            sentence = remove_stopwords(sentence)
            sentence = sentence.split()
            sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence]
            public_key = data[sentence[0]].keys()
            for term in sentence:
                if term in data.keys():
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
        sentence1 = re.sub(punct, " ", split_list[0]) # remove ,
        sentence1 = ' '.join(sentence1.split())
        sentence1 = remove_stopwords(sentence1)
        sentence1 = sentence1.split()
        sentence1 = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence1]
        public_key1 = data[sentence1[0]].keys()
        for term in sentence1:
            if term in data.keys():
                public_key1 = data[term].keys() & public_key1
        sentence1 = ' '.join(sentence1)

        sentence2 = re.sub(punct, " ", split_list[1]) # remove ,
        sentence2 = ' '.join(sentence2.split())
        sentence2 = remove_stopwords(sentence2)
        sentence2 = sentence2.split()
        sentence2 = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence2]
        public_key2 = data[sentence2[0]].keys()
        for term in sentence2:
            if term in data.keys():
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
        sentence = re.sub(punct, " ", split_list[0]) # remove ,
        sentence = ' '.join(sentence.split())
        sentence = remove_stopwords(sentence)
        sentence = sentence.split()
        sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence]
        public_key = data[sentence[0]].keys()
        for term in sentence:
            if term in data.keys():
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
        sentence = re.sub(punct, " ", split_list[1]) # remove ,
        sentence = ' '.join(sentence.split())
        sentence = remove_stopwords(sentence)
        sentence = sentence.split()
        sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in sentence]
        public_key = data[sentence[0]].keys()
        for term in sentence:
            if term in data.keys():
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
    distance = 2
    stemmer = SnowballStemmer("porter")
    punct = re.compile("[^\w\s]") # regax of token
    sentence = re.sub(punct, " ", sentence) # remove ,
    sentence = ' '.join(sentence.split())
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
    stemmer = SnowballStemmer("porter")
    punct = re.compile("[^\w\s]") # regax of token
    sentence = re.sub(punct, " ", sentence) # remove ,
    sentence = ' '.join(sentence.split())
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

def compute_tf(content, document, data):
    terms = content.split()
    tf_list = []
    for t in terms:
        if t in data.keys():
            if str(document) in data[t].keys():
                tf = len(data[t][str(document)])
            else:
                tf = 0.1
            tf_list.append(tf)
    return tf_list, len(tf_list)

def compute_df(content, data):
    terms = content.split()
    df_list = []
    for t in terms:
        if t in data.keys():
            df = len(data[t])
            df_list.append(df)
    return df_list, len(df_list)

def retrival_search(content, data, N):
    stemmer = SnowballStemmer("porter")
    punct = re.compile("[^\w\s]") # regax of token
    content = re.sub(punct, " ", content) # remove ,
    content = ' '.join(content.split())
    content = remove_stopwords(content)
    content = content.split()
    sentence = [stemmer.stem(word) if not word == stemmer.stem(word) else word for word in content]
    # _, sentence = search(content, data)
    public_keys = data[sentence[0]].keys()
    for term in sentence:
        if term in data.keys():
            public_keys = data[term].keys() | public_keys
    sentence = ' '.join(sentence)
    df, df_len = compute_df(sentence, data)

    
    w_matrix = np.zeros((df_len, len(public_keys)))
    # df = np.array(df)[:,None]
    for column, doc in enumerate(public_keys):
        tf, tf_len = compute_tf(sentence, doc, data)

        # tf = np.array(tf)[:,None]
        w_matrix[:,column] = (1 + np.log10(tf))*(np.log10(N)-np.log10(df))
    score_q = np.sum(w_matrix,axis=0)[:,None]
    doc_array = np.array([doc for doc in public_keys])[:,None]
    score_q = np.concatenate((doc_array, score_q), axis=1)
    score_q = score_q[score_q[:,1].argsort()][::-1]
    return score_q

def process_query(query_path):
    punct = re.compile("[^\w\s]") # regax of token
    with open(query_path, 'r', encoding='utf-8') as f:
        querys = f.readlines()
    query_list = []
    for line in querys:
        line = re.split(r"^\d{1,}", line)[1]
        line = re.split(r"$[\n]", line)[0]
        # print(line)
        # line = re.sub(punct, "", line) # no tokenization!!!! for some important punct will be removed
        # print(line)
        line = line.strip()
        query_list.append(line)
    return query_list

def query_search(query_path, data, N, boolean=False):
    for key in data.keys():
        data[key] = data[key][1]
    query_list = process_query(query_path)
    rank_table = []
    for index, query in enumerate(query_list):
        if boolean:
            doc_dic, sentence = search(query, data)
            for docno in doc_dic:
                rank_table.append([index+1, str(docno)])
        else:
            score_q = retrival_search(query, data, N)
            for score_qd in score_q:
                rank_table.append([index+1, str(score_qd[0]), float(score_qd[1])])
    return rank_table

def write_query_results(results, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(str(line)[1:-1]+'\n')

if __name__ == "__main__":
    import time
    trec_path = 'C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\trec.5000.xml'
    index_path = 'C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\index'
    t1 = time.time()
    dic = load_xml(trec_path)
    t2 = time.time()
    print("load_xml+preprocess: {}".format(t2-t1))
    terms = find_uniq(dic)
    print("term numbers: {}".format(len(terms)))
    collection_dic = create_position_index(terms)
    save_position_index(collection_dic, path=index_path) # generate position index

    with open(trec_path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
    # load total text in the dict_file
    dict_file = xmltodict.parse(xml_file)
    dict_file = dict_file['document']['DOC']
    N = len(dict_file) # calculate N

    # testing search
    with open(index_path+'.dat', 'rb') as f:
        data = pickle.load(f)

    boolean_label = True
    t1 = time.time()
    if boolean_label:
        query_path = 'C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\queries.boolean.txt'
        query_results = query_search(query_path, data, N, boolean=True)
        t2 = time.time()
        print("boolean search time: {}".format(t2-t1))
        result_path = 'results.boolean.txt'
        write_query_results(query_results, result_path)
    else: 
        query_path = 'C:\\Users\\mrj\\Desktop\\courses\\TTDS\\CW1\\collections\\queries.ranked.txt'
        query_results = query_search(query_path, data, N, boolean=False)
        t2 = time.time()
        print("rank search time: {}".format(t2-t1))
        result_path = 'results.ranked.txt'
        write_query_results(query_results, result_path)