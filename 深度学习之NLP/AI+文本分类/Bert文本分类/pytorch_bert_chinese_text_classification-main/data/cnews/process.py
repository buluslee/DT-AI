labels = []

with open('cnews.train.txt','r') as fp:
    lines = fp.read().strip().split('\n')
    for line in lines:
        line = line.split('\t')
        labels.append(line[0])

labels = set(labels)
with open('./labels.txt','w') as fp:
  fp.write("\n".join(labels))