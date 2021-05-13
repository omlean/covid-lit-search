import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.sparse import save_npz, load_npz

from scipy.spatial.distance import cosine

import preprocessing

################################################################################################

def tfidf_vectorize(documents, pickle_path=None, save_files_prefix=""):
    """Input:
    documents: Series or List of strings to vectorize
    pickle_path: path of directory to save vectorizer and term-document matrix, e.g. 'data/processed/'
    save_files_prefix: prefix for saved files. For example, passing "01" will produce files '01_vectorizer.pkl' and '01_tdm.npz'
    
    Output: Fit vectorizer and term-document matrix"""
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)
    tdm = vectorizer.transform(documents)
    
    if pickle_path is not None: # save vectorizer and term-document matrix
        
        # if files by that name already exist, prompt user to choose another prefix. Repeats if new input still exists
        while os.path.exists(pickle_path + save_files_prefix + "_vectorizer.pkl"):
            save_files_prefix = input("Files by that name already exist. Enter another prefix...")
        
        vec_path = pickle_path + save_files_prefix + "_vectorizer.pkl"
        
        with open(vec_path, 'wb') as file: # pickle vectorizer
            pickle.dump(vectorizer, file)
        print('Vectorizer pickled at ', vec_path)
        
        tdm_path = pickle_path + save_files_prefix + "_tdm.npz"
        save_npz(tdm_path, tdm) # save term-document matrix
        print('Term-document matrix saved at ', tdm_path)

    return vectorizer, tdm    

################################################################################################

def load_vectorizer(filepath):
    with open(filepath, 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

################################################################################################

# function for cleaning and vectorizing input query
def clean_vectorize_query(query_string, vectorizer):
    """Input: query string (raw), vectorizer fit to search corpus.
    Output: Query vector."""
    clean = preprocessing.clean_text(query_string)
    v = vectorizer.transform([clean]).toarray()
    return v

################################################################################################

def tfidf_search(query, vectorizer, term_document_matrix, index, num_top_results=5):
    """Input:
    query: raw query string
    term_document_matrix: term-document matrix transformed with same vectorizer as the query vector
    index: lookup index for document IDs, e.g. a list or Series. Must return the relevant cord_uid for vector i using `index[i]`
    Output: array of cord_uids for top results."""
    
    v = clean_vectorize_query(query, vectorizer)
    print('Vectorized search query')
    
    num_documents = term_document_matrix.shape[0]
    scores = np.ones(num_documents)
     
    print('Computing document similarity...')
    for i in tqdm(range(num_documents)):
        scores[i] = cosine(v, term_document_matrix[i,:].toarray())
    print('Complete')
    
    top_results = np.argsort(scores)[:num_top_results]
    top_results = np.array(top_results)
    
    uids = [index[i] for i in top_results]
    print(f'Returned top {num_top_results} results.')
    
    return uids

################################################################################################

def write_details(query, uids, reference_df, record_file_prefix, directory='results/'):
    if type(uids) == pd.Series:
        uids = list(uids.values)
    
    record_path = directory + f'{record_file_prefix}_search_record.txt'
    
    while os.path.exists(record_path):
        record_file_prefix = input("Record file by that name already exists; select another:")
        record_path = directory + f'{record_file_prefix}_search_record.txt'
    
    with open(record_path, 'w') as file:
        file.write(f"Query: {query} \n\n")
        for i in range(len(uids)):
            row = reference_df[reference_df.cord_uid == uids[i]].iloc[0]
            file.write(f"Result # {i+1}: cord-uid {uids[i]} \n")
            file.write(f"Title: \n {row.title} \n")
            file.write(f"Abstract: \n{row.abstract}\n\n")
    print(f'Search results saved to {record_path}')
    
################################################################################################