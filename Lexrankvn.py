from lexrank import LexRank
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
import re
import py_vncorenlp

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])
tokenizer = TreebankWordTokenizer()


def word_segment(corpus, split=0):
    py_vncorenlp.download_model()
    documents = []
    for desc in corpus:
        segment = rdrsegmenter.word_segment(desc)
        join_split = re.split(' ', ' '.join(segment))
        filter_punctuation = [x for x in join_split if x.isidentifier()]
        doc = [x.replace('_', ' ').lower() for x in filter_punctuation]
        documents.append([desc, doc])
    if split <= 0 or split > 1:
        return documents
    else:
        train_size = int(len(documents) * split)
        train = documents[:train_size]
        test = documents[train_size:]
        return train, test


def lexrank_vn(train, test=None, sum_size=10, threshold=.1):
    if test is None:
        test = []
    inpt = [x[1] for x in train]
    lexrank_ = LexRank(inpt)
    if not test:
        return lexrank_
    else:
        test_desc = [x[0] for x in test]
        test_inpt = [x[1] for x in test]
        test_outpt = []
        for i in range(len(test)):
            print('- Description:', test_desc[i])
            summary = list(set(lexrank_.get_summary(test_inpt[i], summary_size=sum_size, threshold=threshold)))
            summary_filtered = [x for x in summary if len(x.split()) > 1]
            if summary_filtered:
                test_outpt.append([test_desc[i], summary_filtered])
                print('- Hashtags list:', summary_filtered)
            else:
                test_outpt.append([test_desc[i], summary])
                print('- Hashtags list:', summary)
            print('*'*40)
        return lexrank_, test_outpt


def save_result(out, filename='Lexrank.csv'):
    df = pd.DataFrame(out, columns=['Description', 'Hashtags list'])
    df.to_csv(filename, index=False)
