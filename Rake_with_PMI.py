from rake_nltk import Rake
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

f = open('vietnamese-stopwords.txt')
stopwords_vn = f.read().split('\n')


def rake(desc, ngram=2):
    r = Rake(stopwords=set(stopwords_vn))
    r.extract_keywords_from_text(desc)
    kw = r.get_ranked_phrases()
    if ngram == 2:
        filter_1 = [phrase.lower() for phrase in kw if len(phrase.split()) > 1]
    elif ngram == 3:
        filter_1 = [phrase.lower() for phrase in kw if len(phrase.split()) > 2]
    else:
        raise Exception("We only support Bigram and Trigram for now. Please input ngram=3 for Trigram or omit it")
    final = list(dict.fromkeys(filter_1))
    return final


def pmi(desc, ngram=2):
    tokens = word_tokenize(desc)
    words = [word for word in tokens if word not in set(stopwords_vn)]
    if ngram == 2:
        finder = BigramCollocationFinder.from_words(words)
        pmi_ = finder.score_ngrams(bigram_measures.pmi)
    elif ngram == 3:
        finder = TrigramCollocationFinder.from_words(words)
        pmi_ = finder.score_ngrams(trigram_measures.pmi)
    else:
        raise Exception("We only support Bigram and Trigram for now. Please input ngram=3 for Trigram or omit it")
    pair_scoring = [(' '.join(item[0]).lower(), item[1]) for item in pmi_ if ''.join(item[0]).isalnum()]
    all_pairs = [item[0] for item in pair_scoring]
    return all_pairs, pair_scoring


def get_kw(desc, top_n=-1, ngram=2):
    filtered_kw = rake(desc, ngram)
    all_pair, pair_scoring = pmi(desc, ngram)
    res = []
    for keyw in filtered_kw:
        c = 0.0
        s = 0.0
        for i in range(len(all_pair)):
            if all_pair[i] in keyw:
                c += 1
                s += pair_scoring[i][1]
        try:
            res.append((keyw, s/c))
        except ZeroDivisionError:
            res.append((keyw, 0.0))
    sorted_res = sorted(res, key=lambda x: x[1], reverse=True)
    if top_n <= 0:
        top_res = sorted_res
    else:
        top_res = sorted_res[:top_n]
    final_res = [kw[0] for kw in top_res]
    return final_res
