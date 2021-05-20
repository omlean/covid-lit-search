import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import json

import matplotlib.pyplot as plt

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from gensim.matutils import cossim

from preprocessing import clean_text_lda

class MyCorpus():
    
    def __init__(self, doc_path_list, dictionary=None):
        self.doc_path_list = doc_path_list
        self.dictionary = dictionary
        if self.dictionary is not None:
            self.id2word = self.dictionary.id2token
        
    def __len__(self):
         return len(self.doc_path_list)
        
    def make_dictionary(self, save_directory=None, file_name="_"):
        print("Creating dictionary...")
        self.dictionary = Dictionary(open(doc_path, 'r').read().split() for doc_path in self.doc_path_list)
        print("...complete")
        _ = self.dictionary[0]
        self.id2word = self.dictionary.id2token
        if save_directory is not None:
            self.dict_save_path = save_directory + file_name + '.dict'
            self.dictionary.save(self.dict_save_path)
            print(f"Saving dictionary to {save_directory}")
        
    def trim_dictionary(self, min_freq=0, stopwords=[], load_from_file=False):
        if load_from_file:
            self.dictionary = Dictionary.load(self.dict_save_path)
        once_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.items() if docfreq <= min_freq] if min_freq > 0 else []
        stop_ids = [self.dictionary.token2id[token] for token in stopwords]
        self.dictionary.filter_tokens(stop_ids + once_ids)
        _ = self.dictionary[0]
        self.id2word = self.dictionary.id2token
        self.dictionary.compactify()  # remove gaps in id sequence after words that were removed
        self.dictionary.save(self.dict_save_path)
        print(f"Dictionary updated at {self.dict_save_path}")
        
    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000, load_from_file=False):
        if load_from_file:
            self.dictionary = Dictionary.load(self.dict_save_path)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        self.dictionary.compactify()  # remove gaps in id sequence after words that were removed
        _ = self.dictionary[0]
        self.id2word = self.dictionary.id2token
        self.dictionary.save(self.dict_save_path)
        
    def get_doc_bow(self, path):
        with open(path, 'r') as file:
            text = file.read()
        return self.dictionary.doc2bow(text.split())
        
    def __iter__(self):
        for doc_path in self.doc_path_list:
            with open(doc_path, 'r') as file:
                text = file.read()
            yield self.dictionary.doc2bow(text.split())
            
######################################################################################################

def doc_topic_matrix(corpus, model):
    matrix = np.zeros((len(corpus), len(corpus.dictionary)))
    row = 0
    for bow in tqdm(corpus):
        doc_topics = model[bow]
        for col, value in doc_topics:
            matrix[row,col] = value
        row += 1
    return matrix

######################################################################################################

def query_to_topics(query, dictionary, model):
    """Input: raw string query.
    Output: Predicted topic distribution of query based on model"""
    query_clean = clean_text_lda(query)
    query_vec = dictionary.doc2bow(query_clean.split())
    query_topics = model[query_vec]
    return query_topics

######################################################################################################

def lda_search(query, model, corpus, dictionary, reference_df, num_top_results=5):
    """Input: Search query
    Output: Results of search: Title, Abstract, Date, Link(s)"""
    
    def uid(path):
        return re.findall(r'(\w+)_clean.txt', path)[0]

    query_vector = query_to_topics(query, dictionary, model) # vectorize query string
    distances = np.array([cossim(query_vector, model[document]) for document in tqdm(corpus)]) # create vector of similarities 
    top_indices = np.argsort(distances)[-num_top_results:][::-1] # find n closest documents
    filelist = [corpus.doc_path_list[i] for i in top_indices]
    uids = [uid(file) for file in filelist] # recover uids
    results_table = reference_df[reference_df.cord_uid.apply(lambda x: x in uids)] # recover document details from reference_df
            
    return results_table

######################################################################################################

# check number of topics per document
def topics_per_doc(model, corpus):
    """Plots distribition of number of topics per document"""
    num_topics = []
    for doc in tqdm(corpus):
        pred = model[doc]
        num_topics.append(len(pred))
    plt.hist(num_topics, bins=max(num_topics))
    plt.show()

######################################################################################################

# list top words in topic
def topic_words(topic_no, model, corpus, topn=20):
    ids = model.get_topic_terms(topic_no, topn=50)
    words = [corpus.id2word[i] for i, amt in ids]
    return words

######################################################################################################
######################################################################################################

# Investigation functions

def print_pred(query, corpus, model):
    """Prints details about input query relative to a corpus and model:
    - The cleaned query text
    - The bag of words
    - Query topic predictions
    - Topics for individual words from the query."""
    
    def word_topics(word_id):
        return(model.get_term_topics(word_id, minimum_probability=None))

    print(query, '\n')
    query = clean_text_lda(query)
    print(query, '\n')
    q_vec = corpus.dictionary.doc2bow(query.split())
    print("bow: ")
    for i, n in q_vec:
        print(corpus.id2word[i], n)
    pred = model[q_vec]
    pred.sort(key=lambda x: x[1])
    pred = pred[::-1]
    print("\nTopic predictions: ")
    print(pred, '\n')
    for word, _ in q_vec:
        print(corpus.id2word[word], word_topics(word))
        
######################################################################################################

def uid_pred(uid, model, corpus, metadata):
    """Retrieves predictions for input UID"""
    print(metadata[metadata.cord_uid == uid].title.iloc[0])
    path = [path for path in corpus.doc_path_list if uid in path][0]
    bow = corpus.get_doc_bow(path)
    pred = model[bow]
    pred.sort(key=lambda x: x[1])
    pred = pred[::-1]
    print("\nTopic predictions: ")
    print(pred)
    
######################################################################################################

def search_titles(query, model, corpus, dictionary, metadata, print_titles=True):
    """Run search on the query, returns DataFrame of results. Optionally prints titles."""
    results_df = my_lda.lda_search(query, model, corpus, dictionary, metadata)
    if print_titles:
        for title in results_df.title.values:
            print(title, '\n')
    return results_df

######################################################################################################

def get_body_text_from_path(path):
    """Input:
    path [string]: full path of target file
    Output:
    String of body text"""
    blocks = []
    with open(path, 'r') as file:
        j = json.load(file)
        for block in j['body_text']:
            blocks.append(block['text'])
    fulltext = '\n\n'.join(blocks)
    return fulltext

######################################################################################################

def print_parsed_text(uid, version='raw', clean_directory='data/cord-19/body_text/lda_raw/'):
    """In: uid
    Out: Raw or Cleaned text from """
    if version == 'raw':
        directory = 'data/cord-19/body_text/lda_raw/'
        suffix = '.txt'
    if version == 'clean':
        directory = clean_directory
        suffix = '_clean.txt'
    path = directory + uid + suffix
    with open(path, 'r') as file:
        text = file.read()
    print(text)

######################################################################################################