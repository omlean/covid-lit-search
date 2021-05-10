# import libraries
import pandas as pd
import json
import nltk
import re
import string
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def parse_questions(filepath):
    """Parses the JSON files containing consumer and expert questions from EPIC-QA.
    Returns DataFrame"""
    with open(filepath, 'r') as file:
        j = json.load(file)
    
    question_id, question, query, background = [], [], [], []
    
    for item in j:
        question_id.append(item['question_id'])
        question.append(item['question'])
        query.append(item['query'])
        background.append(item['background'])
        
    df = pd.DataFrame(data={'question_id': question_id,
                        'question': question,
                        'query': query,
                        'background': background})
    
    return df

##############################################################################

def drop_emptier_duplicates(df):
    """For all sets of rows with the same value of duplicate_column, keep only the one with the fewes NaNs"""
    duplicates_df = df[df['cord_uid'].duplicated(keep=False)]
    duplicates_df['nans'] = duplicates_df.apply(lambda x: x.isnull().sum(), axis=1)
    droplist = []
    print("Choosing rows to drop")
    for uid in tqdm(duplicates_df['cord_uid'].unique()):
        sets = duplicates_df[duplicates_df['cord_uid'] == uid]
        for i in sets.sort_values('nans', ascending=False).iloc[1:].index:
            droplist.append(i)
    return df.drop(index=droplist)

##############################################################################

def clean_metadata(filepath):
    """Loads and preprocesses CORD-19 metadata table at the specified location.
    Returns DataFrame"""
    df = pd.read_csv(filepath, low_memory=False)
    print('CSV file loaded successfully')
    
    # drop unwanted columns
    drop_columns = ['sha', 'license']
    df = df.drop(columns=drop_columns)

    # drop rows with no title (these appear to be non-English articles)
    df = df[df.title.notnull()]
    
    # convert publish_time column to datetime format
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    
    print('Dropping duplicate uids')
    df = drop_emptier_duplicates(df)
    
    print('Metadata cleaning complete')
    
    return df

##############################################################################

def load_cleaned_metadata(path):
    # load dataframe
    df = pd.read_csv(path, index_col=0, sep='\t', low_memory=False)
    
    # convert publish_time to datetime format
    df.publish_time = pd.to_datetime(df.publish_time)
    
    # fill nans with "none"
    none_fill = ['pmcid', 'pubmed_id', 'mag_id', 'who_covidence_id', 'arxiv_id', 's2_id',
            'pdf_json_files', 'pmc_json_files']
    for col in none_fill:
        df[col] = df[col].fillna('none')
    
    # fill nans with "unknown"
    unknown_fill = ['doi', 'publish_time', 'authors', 'journal', 'url']
    for col in unknown_fill:
        df[col] = df[col].fillna('unknown')
    
    # fill nans with empty string
    empty_fill = ['abstract']
    for col in empty_fill:
        df[col] = df[col].fillna('')
        
    return df        

##############################################################################

def clean_text(s, stem=False, lemmatize=True, stopword_list=stopwords.words('english')):
    """Removes punctuation, lowercases, removes stopwords, 
    removes digit-only words, stems (optional) and lemmatizes (optional).
    Note: to remove no stopwords, pass [] as stopword_list."""
    s = s.lower() # lowercase
    s = " ".join([w for w in word_tokenize(s) if w not in stopword_list]) # remove stopwords
    s = s.lower() # lowercase
    s = re.sub(r'[-–]', ' ', s) # replace hyphens with spaces
    s = re.sub(r'[^a-z|0-9|\s]', '', s) # remove anything that isn't alphanumeric or whitespace
    s = re.sub(r'\s\d+\s', ' ', s) # remove digit-only words
    
    if stem:
        porter = PorterStemmer()
        stemmed_words = []
        for word in word_tokenize(s):
            stemmed_words.append(porter.stem(word))
        s = ' '.join(stemmed_words)
        
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = []
        for word in word_tokenize(s):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='v'))
        s = ' '.join(lemmatized_words)
    
    # Drops words containing more than 4 digits, or beginning with digits then letters
    s = re.sub(r'\d[a-z]*\d[a-z]*\d[a-z]*\d[a-z]*\d[\d\w]*', '', s)
    s = re.sub(r'\d+[a-z]+\W*', '', s)
    s = " ".join(word.strip() for word in s.split())
    
    return s

##############################################################################

def make_search_documents(df, stem=False, lemmatize=True, stopword_list=stopwords.words('english')):
    """Input: dataframe whose titles and abstracts are to be merged into clean documents for vectorization.
        Must contain columns ['cord_uid', 'title', 'abstract']
    Output: List of cleaned strings consisting of titles and abstracts"""
    l = []
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        s = row.title + " " + row.abstract
        cleaned_text = clean_text(s, stem=stem, lemmatize=lemmatize, stopword_list=stopword_list)
        l.append(cleaned_text)

    return l

##############################################################################