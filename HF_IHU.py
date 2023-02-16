import math
import json


# separating term and hashtag
# input a corpus of lists of tokenized texts
# output: [[hashtag_list_1, term_list_1],
#          [hashtag_list_2, term_list_2],
#          ...
#          [hashtag_list_n, term_list_n]]
def separate_t_and_h(corpus):
    result = []
    for lst in corpus:
        hashtag_list = []
        term_list = []

        # separate terms and hashtags
        for token in lst:
            if token[0] == '#':
                hashtag_list.append(token)
            else:
                term_list.append(token)
        result.append([hashtag_list, term_list])

    return result


# generate hashtag frequency map (HFM)
# the input is the output of separate_t_and_h function
# the output is a dictionary:
# { hashtag1: { term1.1: count, term1.2: count,... },
#   hashtag2: { term2.1: count, term2.2: count,... },... }
def hfm(seperated_corpus, save_name='hfm.json'):
    result = {}

    for lst in seperated_corpus:
        hashtag_list = lst[0]
        term_list = lst[1]

        for hashtag in hashtag_list:
            try:
                for term in term_list:
                    try:
                        result[hashtag][term] += 1
                    except KeyError:
                        result[hashtag][term] = 1
            except KeyError:
                temp_term_counter = {}
                for term in term_list:
                    try:
                        temp_term_counter[term] += 1
                    except KeyError:
                        temp_term_counter[term] = 1
                result[hashtag] = temp_term_counter

    obj = json.dumps(result, indent=2, ensure_ascii=False)
    with open(save_name, 'w') as f:
        f.write(obj)
    return result


# generate term to hashtag-frequency-map (THFM)
# the input is the output of separate_t_and_h function
# the output is a dictionary:
# { term1: { hashtag1.1: count, hashtag1.2: count,... },
#   term2: { hashtag2.1: count, hashtag2.2: count,... } }
def thfm(seperated_corpus, save_name='thfm.json'):
    result = {}

    for lst in seperated_corpus:
        hashtag_list = lst[0]
        term_list = lst[1]

        for term in term_list:
            try:
                for hashtag in hashtag_list:
                    try:
                        result[term][hashtag] += 1
                    except KeyError:
                        result[term][hashtag] = 1
            except KeyError:
                temp_hashtag_counter = {}
                for hashtag in hashtag_list:
                    try:
                        temp_hashtag_counter[hashtag] += 1
                    except KeyError:
                        temp_hashtag_counter[hashtag] = 1
                result[term] = temp_hashtag_counter

    obj = json.dumps(result, indent=2, ensure_ascii=False)
    with open(save_name, 'w') as f:
        f.write(obj)
    return result


# calculate hashtag frequency - inverse hashtag ubiquity (HF-IHU)
# the input are the tokenized list of words, the output of hfm function and thfm function
# the output is a dictionary:
# { hashtag1: hf-ihu1,
#   hashtag2: hf-ihu2,... }
def hf_ihu(inp, hfm_res, thfm_res):
    result = {}
    corpus_length = len(thfm_res)

    for term in inp:
        try:
            term_thfm = thfm_res[term]
            count_h = 0
            for hashtag in term_thfm:
                count_h += term_thfm[hashtag]

            for hashtag in term_thfm:
                hf = term_thfm[hashtag] / count_h  # calculate HF
                ihu = math.log(corpus_length / len(hfm_res[hashtag]))  # calculate IHU
                score = hf * ihu
                # calculate HF-IHU
                try:
                    result[hashtag] += score  # calculate HF-IHU
                except KeyError:
                    result[hashtag] = score
        except KeyError:
            continue
    return result


# sort the result of HF-IHU
# the input is the output of the hf_ihu function
# the output is the sorted output of the hf_ihu function
def sort_score(result):
    sorted_by_value = sorted(result.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_by_value)


def read_json(hfm_filename='hfm.json', thfm_filename='thfm.json'):
    hfm_f = open(hfm_filename, 'r')
    hfm_reader = json.loads(hfm_f.read())
    thfm_f = open(thfm_filename, 'r')
    thfm_reader = json.loads(thfm_f.read())
    return hfm_reader, thfm_reader
