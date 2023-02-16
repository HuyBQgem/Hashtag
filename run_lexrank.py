from data_utils import get_test_data_with_tags
from Lexrankvn import *

threshold = 0.7
data = get_test_data_with_tags('test1.csv')
all_desc = data['description']
train, test = word_segment(all_desc, split=.8)

lxr, out = lexrank_vn(train, test, threshold=threshold)
save_result(out, filename=f'Lexrank_{threshold}.csv')
