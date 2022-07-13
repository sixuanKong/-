import os
from random import shuffle

data_total = [w.strip() for w in open('实验五数据/train.txt', 'r').readlines()][1:]
data_test = [w.strip() for w in open('实验五数据/test_without_label.txt', 'r').readlines()][1:]


shuffle(data_total)
LEN = len(data_total)
trains = data_total[int(LEN / 5):]
valid = data_total[:int(LEN / 5)]
labels = sorted(list(set([w.split(',')[1] for w in data_total])))

f1 = open('image/train.txt', 'w', encoding='utf8')
with open('text/train.txt', 'w', encoding='utf8') as f:
    for t in trains:
        text_index, label = t.split(',')
        label = str(labels.index(label))
        content = open(f'image/data/{text_index}.txt', 'r', encoding='gbk', errors='ignore').read()
        f.write(content.strip() + '\t' + label + '\n')
        f1.write(t + '\n')
    f1.close()

f2 = open('image/dev.txt', 'w', encoding='utf8')
with open('text/dev.txt', 'w', encoding='utf8') as f:
    for v in valid:
        text_index, label = v.split(',')
        label = str(labels.index(label))
        content = open(f'image/data/{text_index}.txt', 'r', encoding='gbk', errors='ignore').read()
        f.write(content.strip() + '\t' + label + '\n')
        f2.write(v + '\n')
    f2.close()

f3 = open('image/test.txt', 'w', encoding='utf8')

with open('text/test.txt', 'w', encoding='utf8') as f:
    for t in data_test:
        text_index, _ = t.split(',')
        label = "0"
        content = open(f'image/data/{text_index}.txt', 'r', encoding='gbk', errors='ignore').read()
        f.write(content.strip() + '\t' + label + '\n')
        f3.write(t.replace('null','negative') + '\n')

    f3.close()

with open('text/class.txt', 'w', encoding='utf8') as f:
    for l in labels:
        f.write(l + '\n')
