from tqdm import tqdm

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

class MyCorpus():
    
    def __init__(self, doc_path_list):
        self.doc_path_list = doc_path_list
        
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