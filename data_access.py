import pandas as pd
import json
import os

def get_pdf(uid, metadata_df, directory='data/cord-19/'):
    """In:
    uid [str]: cord-uid of required file
    metadata_df: DataFrame containing metadata for file
    
    Returns:
    json of required file"""
    
    uid_df = metadata_df[metadata_df.cord_uid == uid]
    pdf = uid_df.iloc[0].pdf_json_files
    if pdf == 'none':
        return 'none'
    if ';' in pdf:
        pdf = pdf.split(';')[0].strip()
    pdf = os.path.join(directory, pdf)
    with open(pdf, 'r') as file:
        pdf_json = json.load(file)
    return pdf_json  

def get_pmc(uid, metadata_df, directory='data/cord-19/'):
    """
    In:
    uid [str]: cord-uid of required file
    metadata_df: DataFrame containing metadata for file
    
    Returns:
    json of required file"""
    
    uid_df = metadata_df[metadata_df.cord_uid == uid]
    pmc = uid_df.iloc[0].pmc_json_files
    if pmc == 'none':
        return 'none'
    if ';' in pmc:
        pmc = pmc.split(';')[0].strip()
    pmc = os.path.join(directory, pmc)
    with open(pmc, 'r') as file:
        pmc_json = json.load(file)
    return pmc_json 

def get_body_text(uid, metadata_df, directory='data/cord-19/'):
    """Input:
    uid [string]: cord-uid of target document
    metadata_df [DataFrame]: DataFrame containing metadata for file"""
    j = get_pdf(uid, metadata_df, directory=directory)
    if j == 'none':
        j = get_pmc(uid, metadata_df, directory=directory)
    blocks = []
    for block in j['body_text']:
        blocks.append(block['text'])
    fulltext = ' '.join(blocks)
    return fulltext

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
    fulltext = ' '.join(blocks)
    return fulltext