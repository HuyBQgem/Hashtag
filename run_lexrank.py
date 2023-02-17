from data_utils import get_test_data_with_tags
from Lexrankvn import *
from evaluater import count_true_false, cos_true_false
import re

threshold = .99
data = get_test_data_with_tags('test1.csv')
all_desc = data['description']
all_tags = data['tag']
tags = [re.sub('#', ' ', x).strip().split(', ') for x in all_tags]

train, test = word_segment(all_desc, split=.8)
train_size = int(len(tags) * .8)
train_tags, test_tags = tags[:train_size], tags[train_size:]

lxr, out = lexrank_vn(train, test, threshold=threshold)
save_result(out, filename=f'Lexrank_{threshold}.csv')

out_tags = [x[1] for x in out]
count_true_false(out_tags, test_tags)
cos_true_false(out_tags, test_tags, threshold=.1)
