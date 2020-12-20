import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from time import *
import math


dtype = torch.FloatTensor
begin_time = time()
window_size = 4
n_step = window_size - 1
n_hidden = 50
batch_size = 1000
valid_size = 1000
iters_num = 6

def preprocess(filename):
    file_open = open(filename, "r")
    file_full_sentences = file_open.readlines()
    file_temp_word = []
    for n in range(len(file_full_sentences)):
        file_temp_word.append(file_full_sentences[n].split())
    file_temp_sentences = []
    for n in range(len(file_full_sentences)):
        file_temp_sentences.append([" ".join(file_temp_word[n][i:i + window_size ]) for i in range(len(file_temp_word[n]) - window_size+1)])
    file_part_sentences = []
    for i in range(len(file_temp_sentences)):
        for j in range(len(file_temp_sentences[i])):
            file_part_sentences.append(file_temp_sentences[i][j])
    return file_part_sentences

def caculate(valid_sentences,valid_size):
    with torch.no_grad():
        sum_loss = 0
        length_valid = int(len(valid_sentences)/valid_size)
        for i in range(length_valid):
            input_valid, target_valid = make_batch(valid_sentences,valid_size,i)
            input_valid_batch = torch.Tensor(input_valid)
            target_valid_batch = torch.LongTensor(target_valid)
            if list(input_valid_batch.size())[0] < valid_size:
                valid_hidden = torch.zeros(1, list(input_valid_batch.size())[0], n_hidden)
            else:
                valid_hidden = torch.zeros(1, valid_size, n_hidden)#.cuda()

            output_valid_batch = model(valid_hidden,input_valid_batch)
            valid_loss = criterion(output_valid_batch, target_valid_batch)
            if list(input_valid_batch.size())[0] < valid_size:
                valid_loss *= list(input_valid_batch.size())[0]
            else:
                valid_loss *= valid_size
            sum_loss += valid_loss
        valid_loss = sum_loss / len(valid_sentences)
        ppl = math.pow(math.e,valid_loss)
    return ppl, valid_loss

train_sentences = preprocess("train.txt")
valid_sentences = preprocess("valid.txt")
word_list = " ".join(train_sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)
eye = np.eye(n_class)
def make_batch(sentences,batch_size,i):
    input_batch = []
    target_batch = []
    length = len(sentences)
    for j in range(batch_size):
        input = []
        if i * batch_size + j < length:
            sen = sentences[i * batch_size + j]
        else:
            break
        sen = sen.split()
        for word in sen[:-1]:
            if word in word_dict:
                input.append(word_dict[word])
            else:
                input.append(word_dict["<unk>"])
        n = sen[-1]
        if n in word_dict:
            target_batch.append(word_dict[n])
        else:
            target_batch.append(word_dict["<unk>"])
        input_batch.append(eye[input])
    return input_batch, target_batch

# to Torch.Tensor
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))

    def forward(self, hidden, X):
        # X : [batch_size, n_step, n_class]
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = torch.mm(outputs, self.W) + self.b # model : [batch_size, n_class]
        return model

model = TextRNN()

LR = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training
print("START TRAINING")
length = int(len(train_sentences)/batch_size)+1
print(length)
for epoch in range(iters_num):
    np.random.shuffle(train_sentences)
    for i in range(length):
        begin = time()
        input_batch_big, target_batch_big = make_batch(train_sentences,batch_size,i)
        input_batch = Variable(torch.Tensor(input_batch_big))
        target_batch = torch.LongTensor(target_batch_big)
        optimizer.zero_grad()
        if list(input_batch.size())[0] < batch_size:
            hidden = Variable(torch.zeros(1, list(input_batch.size())[0], n_hidden))
        else:
            hidden = Variable(torch.zeros(1, batch_size, n_hidden))
        output = model(hidden, input_batch)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()
        print("BATCH:%d" % i, "LOSS:{:f}".format(loss))
        if (i + 1) % 100 == 0:
            ppl, valid_loss = caculate(valid_sentences, valid_size)
            print('Epoch:', '%d' % (epoch + 1), 'valid_ppl =', '{:.6f}'.format(ppl))
            if ppl<250:
                for p in optimizer.param_groups:
                    p['lr'] = 0.001
        end = time()
        print("本次batch时间：%f"%(end-begin))

end_time = time()
run_time = end_time - begin_time
print("SUCCESS")
print("程序运行时间：{:f}".format(run_time))


