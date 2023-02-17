import HF_IHU
import json
import re
import pandas as pd
from data_utils import *
from evaluater import cos_true_false

# # 1) dataset without hashtags, using category as hashtags
#
# f = open('test.json')
# data = json.load(f)
#
# corpus = []  # return [[h_list, t_list], [h_list, t-list],...]
# for film in data:
#     term_list = re.split(r'\W+', film['description'])
#     hashtag_list = re.split(', ', film['raw_category_name'])
#     term_list = [x.lower() for x in term_list]
#     corpus.append([hashtag_list, term_list])
#
# thfm = HF_IHU.thfm(corpus[:450])
# hfm = HF_IHU.hfm(corpus[:450])
# test = corpus[450:]
#
# # 1 sample test
# # hf_ihu = HF_IHU.hf_ihu(corpus[-2][1], hfm, thfm)
# # sort = HF_IHU.sort_score(hf_ihu)
# # print(corpus[-2][0])
# # print(list(sort.keys())[:10])
# # print('*'*30)
# # print(sort.keys())
#
# count_true = 0
# count_false = 0
# for i in range(len(test)):
#     tags = corpus[-i][0]
#     hf_ihu = HF_IHU.hf_ihu(corpus[-i][1], hfm, thfm)
#     sort = HF_IHU.sort_score(hf_ihu)
#     for item in tags:
#         if item in list(sort.keys())[:5]:
#             count_true += 1
#         else:
#             count_false += 1
#     print("Ground truth:", corpus[-i][0])
#     print("Prediction:  ", list(sort.keys())[:5])
#     print('*'*50)
#
# print("count true:", count_true)
# print("count false:", count_false)
# # print(count_true / (count_false + count_true))
# # end 1)


# # 2) imbd dataset with hashtags
# tagged_data, no_tag_data = get_imdb_data()
# corpus = imdb_extractor(tagged_data)
# name = 'imdb'

# 3) test with tags dataset
data = get_test_data_with_tags('test1.csv')
corpus = test_data_with_tags_extractor(data)
name = 'hand_tagging'

train_size = int(len(corpus) * .8)
thfm = HF_IHU.thfm(corpus[:train_size], save_name=f'{name}_thfm.json')
hfm = HF_IHU.hfm(corpus[:train_size], save_name=f'{name}_hfm.json')
test = corpus[train_size:]
test_tags = [x[0] for x in test]

out_tags = []
count_true = 0
count_false = 0
try:
    for i in range(len(test)):
        tags = test[i][0]
        hf_ihu = HF_IHU.hf_ihu(test[i][1], hfm, thfm)
        sort = HF_IHU.sort_score(hf_ihu)
        out_tags.append(list(sort.keys())[:len(tags)+2])
        for item in tags:
            if item in list(sort.keys())[:len(tags)+2]:
                count_true += 1
            else:
                count_false += 1

        # print("Description:", test[i][1])
        print("Ground truth:", test[i][0])
        # print("Prediction:  ", list(sort.keys())[:max(5, len(tags))])
        print("Prediction:  ", list(sort.keys())[:len(tags)+2])
        print('*'*50)
except KeyboardInterrupt:
    pass

print("count true:", count_true)
print("count false:", count_false)
print("precision:", count_true * 100 / (count_true + count_false))
cos_true_false(out_tags, test_tags)
# end 2) and 3)


# # 4) data_900 dataset, test has no tag, lack data for corpus
# tagged, no_tag = get_data_900_data()
# corpus = data_900_extractor(tagged)
#
# thfm = HF_IHU.thfm(corpus)
# hfm = HF_IHU.hfm(corpus)
# test = no_tag['description'].reset_index(drop=True)
#
# try:
#     for i in range(len(test)):
#         hf_ihu = HF_IHU.hf_ihu(test[i], hfm, thfm)
#         sort = HF_IHU.sort_score(hf_ihu)
#         print("Ground truth:", test[i])
#         print("Prediction:  ", list(sort.keys())[:5])
#         print('*'*50)
# except KeyboardInterrupt:
#     pass
# # end 4)
