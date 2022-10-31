import re
import time
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
def preprocess_small_file(input_file, output_file):
    # define the regax
    punct = re.compile("[^\w\s]") # regax of token
    input_file = input_file
    output_file = output_file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = re.sub(punct, "", content) # tokenisation
#    new_content = new_content.lower() # make all text into lower case
    new_content = remove_stopwords(new_content) # use gensim to remove sw
    new_content = re.split('\s+', new_content) # convert string to list for stemming; or can use string.split()
    ps = PorterStemmer()
    new_content = [ps.stem(word) if not word == ps.stem(word) else word for word in new_content] # stemming(include lower case)
    new_content = ' '.join(new_content) # convert list to string for writing into file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

def preprocess_big_file(input_file, output_file):
    # read and write once a line
    # define the regax
    punct = re.compile("[^\w\s]") # regax of token
    input_file = input_file
    output_file = output_file
    ps = PorterStemmer()
    with open(input_file, 'r', encoding='utf-8') as f_read:
        with open(output_file, 'w', encoding='utf-8') as f_write:
            for line in f_read:
                line = line.strip() # removing whitespace in the beginning and end of the line
                line = re.sub(punct, "", line) # tokenisation
#                line = line.lower() # make all text into lower case
                line = remove_stopwords(line) # use gensim to remove sw
                line = re.split('s+', line) # convert string to list for stemming; or can use string.split()
                line = [ps.stem(word) if not word == ps.stem(word) else word for word in line] # stemming(include lower case)
                line = ' '.join(line) # convert list to string for writing into file
                line += "\n"
                f_write.write(line)

if __name__=="__main__":
    start1 = time.time()
    preprocess_small_file(input_file="./bib.txt", output_file="bib_preprocess.txt")
    end1 = time.time()
    print("Time for preprocess small file: bib -> {}s".format(end1-start1))
    preprocess_small_file(input_file="./quran.txt", output_file="quran_preprocess.txt")
    end2 = time.time()
    print("Time for preprocess small file:quran -> {}s".format(end2-end1))
    preprocess_small_file(input_file="./wiki.txt", output_file="wiki_preprocess.txt")
    end3 = time.time()
    print("Time for preprocess big file:wiki -> {}s".format(end3-end2))