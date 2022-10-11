import numpy as np

def load_file_labels(file_name):
    texts = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            text = line.strip().lower().split(' ')
            texts.extend(text)
    return texts


ngram = load_file_labels('test.out')
dblstm = load_file_labels('dblstm_test.out')

sum = 0
for i in range(len(ngram)):
    if ngram[i] == dblstm[i]:
        sum += 1
print(sum / len(ngram))

