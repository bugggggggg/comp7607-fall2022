import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import nltk

import copy
import random
import time

from urllib3 import encode_multipart_formdata

def message(msg):
    print(msg, '-'*100)

def set_seed(seed=1120):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### const
PREP_LABEL = '<prep>'
UNKNOWN_LABEL = '<unk>'
PAD_LABEL = '<pad>'
PREP = ['at', 'in', 'of', 'for', 'on']
PREP_MAP = {'at': 0, 'in': 1, 'of': 2, 'for': 3, 'on':4}
CLASSES_NUM = len(PREP)
MAX_LEN = 0

### hyperparameter
SHUFFLE_DATASET = True
BATCH_SIZE = 64
WORD_DIM = 512
LEARNING_RATE=0.01
EPOCH = 20


def load_file_inputs(file_name):
    texts = []
    cnts = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            text = line.strip().lower().split(' ')
            cnt = 0
            for i, word in enumerate(text):
                if word == PREP_LABEL.lower():
                    text[i] = UNKNOWN_LABEL
                    cnt += 1
            cnts.append(cnt)
            for i, word in enumerate(text):
                if word == UNKNOWN_LABEL:
                    tmp = copy.deepcopy(text)
                    tmp[i] = PREP_LABEL
                    texts.append(tmp)
    return texts, cnts

def load_file_labels(file_name):
    texts = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            text = line.strip().lower().split(' ')
            texts.extend(PREP_MAP[prep] for prep in text)
    return np.array(texts)

data_inputs, _ = load_file_inputs('data/prep/dev.in')
data_labels = load_file_labels('data/prep/dev.out')

test_inputs, test_cnts = load_file_inputs('data/prep/test.in')

# print(len(data_inputs))
# print(len(data_labels))
assert(len(data_inputs) == len(data_labels))

message('finish loading')

def add_pad(texts, init_maxlen=0):
    max_len = init_maxlen
    ret = []
    for text in texts:
        p = text.index(PREP_LABEL)
        max_len = max(max_len, p, len(text) - p - 1)
    for i, text in enumerate(texts):
        p = text.index(PREP_LABEL)
        l, r = p, len(text) - p - 1
        tmp = [PAD_LABEL] * (max_len - l) + text + [PAD_LABEL] * (max_len - r)
        tmp[max_len+1:len(tmp)] = tmp[len(tmp)-1:max_len:-1]
        ret.append(tmp)
    return ret, max_len

def init_vocab(texts):
    word2id = {}
    word2id[PAD_LABEL] = 0
    word2id[UNKNOWN_LABEL] = 1
    word2id[PREP_LABEL] = 2
    for text in texts:
        for word in text:
            if word not in word2id:
                word2id[word] = len(word2id)
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2id), WORD_DIM))  ## reverse the second half
    embeddings[word2id[PAD_LABEL]] = np.zeros((WORD_DIM, )) ## <pad> is zero

    for i, text in enumerate(texts):
        texts[i] = [word2id[word] for word in text]
    texts = np.array(texts)
    return word2id, embeddings,

### load test data
test_inputs, test_maxlen = add_pad(test_inputs)
message(f'test_maxlen: {test_maxlen}')

### load train data
data_inputs, MAX_LEN = add_pad(data_inputs, test_maxlen)
word2id, embeddings = init_vocab(data_inputs)
message(f'max_len: {MAX_LEN}')

for i, text in enumerate(test_inputs):
    for j, word in enumerate(text):
        if word in word2id:
            test_inputs[i][j] = word2id[word]
        else:
            test_inputs[i][j] = word2id[UNKNOWN_LABEL]


### split input into train and dev 
def data_loader(inputs_train, inputs_dev, labels_train, labels_dev, batch_size=BATCH_SIZE):
    # print(inputs_train)
    inputs_train = torch.tensor(inputs_train)
    inputs_dev = torch.tensor(inputs_dev)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    labels_dev = torch.tensor(labels_dev, dtype=torch.long)

    data_train = TensorDataset(inputs_train, labels_train)
    sampler_train = RandomSampler(data_train)
    dataloader_train = DataLoader(data_train, sampler=sampler_train, batch_size=batch_size)

    data_dev = TensorDataset(inputs_dev, labels_dev)
    sampler_dev = SequentialSampler(data_dev)
    dataloader_dev = DataLoader(data_dev, sampler=sampler_dev, batch_size=batch_size)

    return dataloader_train, dataloader_dev


inputs_train, inputs_dev, labels_train, labels_dev = train_test_split(\
    data_inputs, data_labels, test_size=0.1, shuffle=SHUFFLE_DATASET)

dataloader_train, dataloader_dev = data_loader(inputs_train, inputs_dev, labels_train, labels_dev, BATCH_SIZE)

message('finish data')




### model #######################################################################
HIDDEN_LAYER_SIZE = 256
vocab_size = len(word2id)

class LSTM(nn.Module):
    def __init__(self, 
                word_dim=WORD_DIM):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=word_dim, hidden_size=HIDDEN_LAYER_SIZE, num_layers=1, batch_first=True)

    def forward(self, input):
        output, (h_n, c_n) = self.lstm(input)
        # print(h_n)
        # print(output.shape)
        return h_n[-1]
        # return output[:,-1,:]

class Double_LSTM(nn.Module):
    def __init__(self, 
                vocab_size,
                word_dim=WORD_DIM,
                pretrained_embedding=None,
                classes_num=CLASSES_NUM,
                dropout=0.5):
        super(Double_LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=word_dim,
                                        max_norm=5.0,
                                        padding_idx=0)
        self.lstm1 = LSTM(word_dim)
        self.lstm2 = LSTM(word_dim)
        self.fc = nn.Linear(2 * HIDDEN_LAYER_SIZE, classes_num)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = self.embedding(input)
        # print(x)
        xl, _, xr = x.split([MAX_LEN, 1, MAX_LEN], dim=1)
        xl = self.lstm1(xl)
        xr = self.lstm2(xr)
        # print(xl)
        x_fc = torch.cat([xl, xr], dim=1)
        x_fc = self.dropout(x_fc)
        # print(x_fc.shape)
        # x_fc = self.dropout(x_fc)
        x_fc = self.fc(x_fc)
        return x_fc


### train ############################################################################
message('start train')

loss_f = nn.CrossEntropyLoss()



def train(model, optimizer, dataloader_train, dataloader_dev, epochs=EPOCH):
    best_accuracy = 0
    for epoch_i in range(epochs):
        start_time = time.time()
        total_error = 0

        model.train()
        for step, batch in enumerate(dataloader_train):
            inputs, labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            y = model(inputs)
            # print(y)
            # exit()
            error = loss_f(y, labels)
            total_error += error.item()
            # print(error)

            error.backward()
            optimizer.step()

        dev_accuracy = evaluate(model, dataloader_dev)
        avg_error = total_error / len(dataloader_train)

        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            torch.save(model.state_dict(), 'double_lstm.pt')
        
        time_elapsed = time.time() - start_time
        print(f'time{time_elapsed:.2f}, epoch_{epoch_i}: dev_accuracy {dev_accuracy:.2f}, loss {avg_error}')

def evaluate(model, dataloader_dev):
    model.eval()
    accuracy_list = []
    print_first = True
    for step, batch in enumerate(dataloader_dev):
        inputs, labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            y = model(inputs)
        preds = torch.argmax(y, dim=1).flatten()
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        accuracy_list.append(accuracy)
        # if print_first:
        #     print(f'labels{labels}')
        #     print(f'predic{preds}')
        #     print_first = False
        # exit(0)

    return np.mean(accuracy_list)


double_lstm = Double_LSTM(len(word2id))
double_lstm.to(device)

# optimizer = optim.Adadelta(double_lstm.parameters(),  ## get all 2 if use Adadelta
#                                 lr=LEARNING_RATE,
#                                 rho=0.95)
optimizer = optim.Adam(double_lstm.parameters(), 
                        lr=LEARNING_RATE)

train(double_lstm, optimizer, dataloader_train, dataloader_dev)
exit()



### model on test data ##########################################################
double_lstm.load_state_dict(torch.load('double_lstm.pt'))
test_inputs = torch.tensor(test_inputs)
# for i in range(len(test_inputs)):
#     test_inputs[i].to(device)
# test_inputs.to(device)

test_labels = double_lstm(test_inputs)
test_labels = torch.argmax(test_labels, dim=1).flatten()

preps = []
p = 0
for cnt in test_cnts:
    tmp = []
    s = p
    while p < s + cnt:
        tmp.append(PREP[test_labels[p].item()])
        p += 1
    preps.append(tmp)

with open('dblstm_test.out', 'w', encoding='utf-8') as f:
    for prep in preps:
        f.write(' '.join(prep))
        f.write('\n')


