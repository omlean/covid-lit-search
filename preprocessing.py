import pandas as pd
import json

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
