from Rake_with_PMI import get_kw
from evaluater import count_true_false, cos_true_false
import pandas as pd
import time
import re
start = time.time()

# cleaned data
data = pd.read_csv('test1.csv')
all_desc = data['description']
all_tags = data['tag']
tags = [re.sub('#', ' ', x).strip().split(', ') for x in all_tags]

res = []
ngram = 3
top_n = 0
for desc in all_desc:
    kw_list = get_kw(desc, top_n=top_n, ngram=ngram)
    res.append([desc, kw_list])
    print('- Description:', desc)
    print('- Keywords list:', kw_list)
    print('*'*30)

df = pd.DataFrame(res, columns=['Description', 'Hashtags list'])
gram = "bi" if ngram == 2 else "tri"
name = f'Rake_with_PMI_top{top_n}_{gram}gram.csv' if top_n > 0 else f'Rake_with_PMI_all_{gram}gram.csv'
df.to_csv(name, index=False)

out_tags = [x[1] for x in res]
count_true_false(out_tags, all_tags)
cos_true_false(out_tags, all_tags, threshold=0.1)
print('Finished in:', time.time() - start)
