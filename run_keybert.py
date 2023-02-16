from KeyBERT import cos_sim_only, max_sum_similarity, maximal_marginal_relevance, embedding
from data_utils import get_imdb_data
import numpy as np
import pandas as pd
import time
import warnings
start = time.time()


def warn(*args, **kwargs):
    pass


def func(inpt, stopword):
    warnings.warn = warn()
    final_res = []
    try:
        for item in inpt:
            res = [item]
            nr = int(len(np.unique(item.split())) / 5)
            top_n = 5
            top_n_mss = top_n if nr >= top_n else nr
            print('- Description: ', item)
            for i in range(1, 4):
                print(f'- Hashtag length = {i}:')
                n_gram_range = (i, i)
                doc_embedding, candidate_embeddings, candidates = embedding(item, n_gram_range, stopword)
                cos = cos_sim_only(doc_embedding,
                                   candidate_embeddings,
                                   candidates,
                                   top=top_n)
                print('- Cosine similarity only:\n', cos)
                mss = max_sum_similarity(doc_embedding,
                                         candidate_embeddings,
                                         candidates,
                                         top=top_n_mss,
                                         nr_candidates=nr)
                print('- Max sum similarity:\n', mss)
                mmr = maximal_marginal_relevance(doc_embedding,
                                                 candidate_embeddings,
                                                 candidates,
                                                 top=top_n,
                                                 diversity=.2)
                print('- Maximum marginal relevance:\n', mmr)
                res.extend([cos, mss, mmr])
            final_res.append(res)
            print('*'*50)
    except Exception as e:
        print(repr(e))

    cols_list = ['Description', 'cos_1', 'mss_1', 'mmr_1', 'cos_2', 'mss_2', 'mmr_2', 'cos_3', 'mss_3', 'mmr_3']
    df = pd.DataFrame(final_res, columns=cols_list)
    df.to_csv('KeyBERT_result.csv', index=False)
    print('Finished in:', time.time() - start)


# # imdb data
# stop_words = "english"  # "english" or a list of stop words corresponding the language within the data
# tagged, _ = get_imdb_data()
# sample = tagged['overview'][:20]
# func(sample, stop_words)

# data1 data
f = open('vietnamese-stopwords.txt')
stop_words_vn = f.read().split('\n')
data = pd.read_csv('data1.csv')
sample = data['description']
func(sample, stop_words_vn)
