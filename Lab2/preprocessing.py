import re
import time
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
def preprocess(input_file, output_file):
    # define the regax
    punct = re.compile("[^\w\s]") # regax of token
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = re.sub(punct, "", content) # tokenisation
    # new_content = new_content.lower() # make all text into lower case
    new_content = remove_stopwords(new_content) # use gensim to remove sw
    new_content = re.split('\s+', new_content) # convert string to list for stemming; or can use string.split()
    ps = PorterStemmer()
    new_content = [ps.stem(word) if not word == ps.stem(word) else word for word in new_content] # stemming(include lower case)
    new_content = ' '.join(new_content) # convert list to string for writing into file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

def extract_text(input_file, output_file):
    # extract ID and text from txt
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.readline
    

if __name__=="__main__":
    start1 = time.time()
    preprocess(input_file=".\\Lab2\\collections\\trec.sample.txt", output_file=".\\Lab2\\collections\\trec.sample_preprocess.txt")
    end1 = time.time()
    print("Time for preprocess small file: bib -> {}s".format(end1-start1))
    