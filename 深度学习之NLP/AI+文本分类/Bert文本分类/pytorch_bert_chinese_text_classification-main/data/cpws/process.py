filename = 'train_data.txt'
labels = set()
with open(filename, 'r', encoding='utf-8') as f:
  raw_data = f.readlines()
  for d in raw_data:
      d = d.strip()
      d = d.split("\t")
      if len(d) == 2:
          labels.add(d[0])

with open('../final_data/labels.txt', 'w', encoding='utf-8') as f:
  f.write("\n".join(list(labels)))