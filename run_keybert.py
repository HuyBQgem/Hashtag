from KeyBERT import cos_sim_only, max_sum_similarity, maximal_marginal_relevance, embedding
from data_utils import get_imdb_data
from evaluater import count_true_false, cos_true_false
import numpy as np
import pandas as pd
import time
import warnings
import re
start = time.time()


def warn(*args, **kwargs):
    pass


def flattening(origin):
    flatten = []
    for item in origin:
        for sub_item in item:
            flatten.append(sub_item)
    return flatten


def func(inpt, all_hashtags, stopword):
    warnings.warn = warn()
    final_res = []
    try:
        for item in inpt:
            res = [item]
            nr = int(len(np.unique(item.split())) / 5)
            top_n = 5
            top_n_mss = top_n if nr >= top_n else nr
            print('- Description: ', item)
            for i in range(2, 3):
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
                                                 diversity=.5)
                print('- Maximum marginal relevance:\n', mmr)
                res.extend([cos, mss, mmr])
            final_res.append(res)
            print('*'*50)
    except Exception as e:
        print(repr(e))
    except KeyboardInterrupt:
        pass

    out_tags = [x[1:] for x in final_res]
    out_tags = [list(set(flattening(x))) for x in out_tags]
    count_true_false(out_tags, all_hashtags)
    cos_true_false(out_tags, all_hashtags)

    # cols_list = ['Description', 'cos_1', 'mss_1', 'mmr_1', 'cos_2', 'mss_2', 'mmr_2', 'cos_3', 'mss_3', 'mmr_3']
    # df = pd.DataFrame(final_res, columns=cols_list)
    # df.to_csv('KeyBERT_result.csv', index=False)
    print('Finished in:', time.time() - start)


# # imdb data
# stop_words = "english"  # "english" or a list of stop words corresponding the language within the data
# tagged, _ = get_imdb_data()
# sample = tagged['overview'][:20]
# func(sample, stop_words)

# data1 data
f = open('vietnamese-stopwords.txt')
stop_words_vn = f.read().split('\n')
data = pd.read_csv('test1.csv')
all_desc = data['description']
all_tags = data['tag']
tags = [re.sub('#', ' ', x).strip().split(', ') for x in all_tags]

func(all_desc, tags, stop_words_vn)
