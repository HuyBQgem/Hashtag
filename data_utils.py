import pandas as pd
import re
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()


def get_imdb_data():
    keywords = pd.read_csv('keywords.csv')
    meta_data = pd.read_csv('movies_metadata.csv')
    # special data cleaning - exclusive
    frostbite_index = meta_data[meta_data['popularity'] == 'Beware Of Frost Bites'].index
    meta_data['popularity'] = pd.to_numeric(meta_data.drop(frostbite_index)['popularity'])
    # merge datas
    merged = pd.concat([meta_data, keywords], axis=1)
    merged = merged.dropna(subset=['popularity', 'overview', 'keywords'])
    # get only the necessaries columns
    data_2_cols = merged[['overview', 'keywords']]
    tagged_data = data_2_cols[data_2_cols['keywords'] != '[]'].reset_index(drop=True)  # data with keywords
    no_tag_data = data_2_cols[data_2_cols['keywords'] == '[]'].reset_index(drop=True)  # data without keywords
    return tagged_data, no_tag_data


# input tagged_data
def imdb_extractor(data):
    corpus = []
    for i in range(len(data)):
        description = data.iloc[i]['overview']
        hashtags = data.iloc[i]['keywords']
        tokenized_desc = tokenizer.tokenize(description)
        rm_punc = [x for x in tokenized_desc if x.isidentifier()]
        tags_list = [x[1:-2] for x in re.findall(r"'[\w\s]+'[^:]", hashtags)]
        corpus.append([tags_list, rm_punc])
    return corpus


def get_data_900_data():
    dataset = pd.read_excel('data_900.xlsx')
    data_2_cols = dataset[['description', 'hash-tag']]
    tagged_data = data_2_cols[:120]
    no_tag_data = data_2_cols[120:]
    return tagged_data, no_tag_data


# input tagged_data of get_data_900_data()
def data_900_extractor(data):
    corpus = []
    for i in range(len(data)):
        description = data.iloc[i]['description']
        hashtags = data.iloc[i]['hash-tag']
        tokenized_desc = tokenizer.tokenize(description)
        rm_punc = [x for x in tokenized_desc if x.isidentifier()]
        tags_list = [x for x in re.split(', ', hashtags)]
        corpus.append([tags_list, rm_punc])
    return corpus


def get_test_data_with_tags(name='test.csv'):
    data = pd.read_csv(name)
    data_2_cols = data[['description', 'tag']]
    return data_2_cols


# input data of get_test_data_with_tags()
def test_data_with_tags_extractor(data):
    corpus = []
    for i in range(len(data)):
        description = data.iloc[i]['description']
        hashtags = data.iloc[i]['tag']
        tokenized_desc = tokenizer.tokenize(description)
        rm_punc = [x for x in tokenized_desc if x.isidentifier()]
        tags_list = [x for x in re.split(',?#', hashtags)][1:-1]
        corpus.append([tags_list, rm_punc])
    return corpus


def get_tags_and_texts(data):
    res = []
    for i in range(len(data)):
        desc = data.iloc[i]['overview']
        tags = data.iloc[i]['keywords']
        tags_list = [x[1:-2] for x in re.findall(r"'[\w\s]+'[^:]", tags)]
        res.append([tags_list, desc])
    return res
