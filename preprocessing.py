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

def load_metadata(filepath):
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
    
    print('Metadata cleaning complete')
    
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

def clean_alphanumeric(s): # now built into clean_text - redundant
    """Drops words containing more than 4 digits, or beginning with digits then letters"""
    s = re.sub(r'\d[a-z]*\d[a-z]*\d[a-z]*\d[a-z]*\d[\d\w]*', '', s)
    s = re.sub(r'\d+[a-z]+\W*', '', s)
    s = " ".join(word.strip() for word in s.split())
    return s

##############################################################################

def make_title_abstract_documents(df, stem=True, lemmatize=True, stopword_list=stopwords.words('english')):
    """Input: DataFrame containing columns ['cord_uid', 'title', 'abstract']
    Returns: Dictionary of cord_uid and cleaned string of merged title and abstract."""
    l = []
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        s = str(row.title) + " " + str(row.abstract) if type(row.abstract) == str else str(row.title)
        l.append(clean_text(s))
    df['title_abstract'] = l
    df = df.drop(columns=['title', 'abstract'])
    return df

##############################################################################