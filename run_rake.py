from Rake_with_PMI import get_kw
import pandas as pd
import time

start = time.time()

# cleaned data
data = pd.read_csv('data1.csv')
all_desc = data['description']

res = []
ngram = 2
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
print('Finished in:', time.time() - start)
