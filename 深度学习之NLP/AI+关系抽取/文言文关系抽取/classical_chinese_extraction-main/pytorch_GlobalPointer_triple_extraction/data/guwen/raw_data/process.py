import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd

if not os.path.exists('../mid_data'):
  os.mkdir("../mid_data")

predicates = set()

with open('train.json', 'r', encoding='utf-8') as fp:
  data = fp.readlines()

count_lengths = []
count_predicates = defaultdict(int)
for d in tqdm(data, ncols=100):
  d = json.loads(d)
  text = d['text']
  spo = d['spo_list']
  predicates.add(spo['predicate'])
  count_predicates[spo['predicate']] += 1
  count_lengths.append(len(text))

with open('../mid_data/predicates.json', 'w', encoding='utf-8') as fp:
  json.dump(list(predicates), fp, ensure_ascii=False)

lengths = Counter(count_lengths)
print(lengths)
print(count_predicates)

