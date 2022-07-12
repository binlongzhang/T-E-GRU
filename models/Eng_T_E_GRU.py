import sys
import numpy as np
import torch
import torch.nn as nn
import re
import pandas as pd
from gensim.models import KeyedVectors
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import random_split,DataLoader
import warnings
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from nltk import tokenize

warnings.filterwarnings("ignore")

# tokenization & get sequence
def getdata(filename, num_words=50000, max_tokens=90):
    data = pd.read_csv(filename)
    data = data.to_numpy()

    for item in data:
        text = re.sub("<br />|\'s|\'", "", item[0])
        tokens = tokenize.word_tokenize(text)
        cut_list = list(range(len(tokens)))
        for i, word in enumerate(tokens):
            try:
                cut_list[i] = cn_model.vocab[word].index
            except:
                cut_list[i] = cn_model.vocab['unk'].index
        item[0] = np.array(cut_list)

    train_pad = pad_sequences(data[:, 0], maxlen=max_tokens, padding='pre', truncating='pre',value=cn_model.vocab['.'].index)
    data_set = [(train_pad[i], data[i][1]) for i in range(len(train_pad))]
    return data_set

def embedding_matrix(num_words = 50000,embedding_dim = 300):

    # 初始化embedding_matrix
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for i in range(num_words):
        embedding_matrix[i,:] = cn_model[ cn_model.index2word[i] ]
    embedding_matrix = embedding_matrix.astype('float32')
    return embedding_matrix


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1,num_words = 50000,T_E_dropout=0.3,RNNDropout=0,T_E_hidden=2048,head_Num=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix(num_words)))
        self.embedding.requires_grad = False

        self.attention = nn.TransformerEncoderLayer(300,head_Num,T_E_hidden,dropout=T_E_dropout)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,dropout=RNNDropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.attention(x)
        output,h = self.rnn(x, h)
        output = self.linear(output[-1])
        output = self.softmax(output)
        return output

    def initHidden(self, batch_size):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h_0




def train(train_db, net, batch_size=20):
    net.train()
    train_loss = 0
    train_acc = 0

    data = DataLoader(train_db, batch_size=batch_size, shuffle=True,drop_last=True)
    epoch = 0
    for i, (text, label) in enumerate(data):
        optimizer.zero_grad()

        text = text.long().to(device)
        label = label.long().to(device)

        h = net.initHidden(text.shape[0]).to(device)
        output = net(text, h)
        loss = criterion(output, label)

        train_acc += (label.view(-1, 1) == output.topk(1)[1]).sum().item()
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        epoch = epoch + 1

    return train_loss / (epoch*batch_size), train_acc / (epoch*batch_size)

def valid(val_db, net, batch_size=20):
    net.eval()
    val_loss = 0
    val_acc = 0

    data = DataLoader(val_db, batch_size=batch_size, shuffle=True)
    epoch = 0
    for text, label in data:
        with torch.no_grad():
            text = text.long().to(device)
            label = label.long().to(device)
            h = net.initHidden(text.shape[0]).to(device)

            output = net(text, h)
            loss = criterion(output, label)

            val_acc += (label.view(-1, 1) == output.topk(1)[1]).sum().item()
            val_loss += loss.item()
            epoch = epoch+1

    return val_loss / (epoch*batch_size), val_acc / (epoch*batch_size)

def test(test_db, net, batch_size=20):
    net.eval()
    data = DataLoader(test_db, batch_size=batch_size)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for text, label in data:
        with torch.no_grad():
            text = text.long().to(device)
            label = label.long().to(device)
            h = net.initHidden(text.shape[0]).to(device)

            output = net(text, h)
            prediction = output.topk(1)[1]

            for p, t in zip(prediction.view(-1), label.view(-1)):
                if((p==1) & (t==1)):
                    TP = TP + 1
                elif((p==0) & (t==0)):
                    TN = TN + 1
                elif((p==1) & (t==0)):
                    FP = FP +1
                elif((p==0) & (t==1)):
                    FN = FN +1
    return TP,TN,FP,FN


print('-------加载词嵌入模型---------')
# gensim  load word embedding
cn_model = KeyedVectors.load_word2vec_format( './embeddings/glove-wiki-gigaword-300.gz')


if __name__ == '__main__':

    PathRoot = '../'
    dataName = 'IMDB'
    T_E_hidden = 2048
    GRU_hiddenSize = 256
    head_Num = 2
    RNNDropout = 0
    T_E_dropout = 0.3
    netName = 'T-E-GRU-' + str(head_Num) + '-' + str(T_E_hidden) + '-' + str(GRU_hiddenSize) + '-' + str(T_E_dropout) + '-' + str(RNNDropout)+ str(500)

    writer = SummaryWriter(log_dir=PathRoot +'log/'+dataName+'/'+ netName)

    print('-------初始化网络---------')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Batch_size = 128
    N_EPOCHS = 100
    sequence_len = 500
    num_words = len(cn_model.vocab)

    net = Net(input_size=300, hidden_size=GRU_hiddenSize, output_size=2, batch_size=Batch_size,num_words=num_words)
    # 装载之前训练的部分继续训练
    # net.load_state_dict(torch.load(PathRoot+'log/'+dataName + '/'+netName+'/100.pkl'))

    net = net.to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    writer.add_graph(net, (torch.zeros(Batch_size, sequence_len).long().to(device), net.initHidden(Batch_size).to(device)))
    print('网络结构为：')
    print(net)

    print('-------加载数据---------')
    train_db = getdata(filename=PathRoot+'data/'+dataName+'/train.csv', num_words=num_words, max_tokens=sequence_len)
    val_db = getdata(filename=PathRoot+'data/'+dataName+'/val.csv', num_words=num_words, max_tokens=sequence_len)
    print('-------训练开始---------')
    print('-------运算设备为：' + str(device) + '---------')

    start_time = time.time()
    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(train_db, net, Batch_size)
        valid_loss, valid_acc = valid(val_db, net, Batch_size)
        scheduler.step()

        secs = int(time.time() - start_time)

        mins = secs / 60
        secs = secs % 60

        writer.add_scalars('Loss', {'train': train_loss, 'test': valid_loss}, epoch)
        writer.add_scalars('Acc', {'train': train_acc, 'test': valid_acc}, epoch)

        print('Epoch: %d' % (epoch + 1), " | time in %d minites, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        if ((epoch + 1) % 2 == 0) & ((epoch+1) > (N_EPOCHS-10)):
            torch.save(net.state_dict(), '../log/' +dataName+'/'+ netName + '/' + str(epoch + 1) + '.pkl')
    writer.close()

    print('-------模型测试中---------')
    test_db = getdata(filename=PathRoot + 'data/'+dataName+'/test.csv', num_words=num_words, max_tokens=sequence_len)
    startTestTime = time.time()
    TP, TN, FP, FN = test(test_db, net, Batch_size)
    testTime = time.time() - startTestTime
    print('测试时间：' + str(testTime))
    print('TP:' + str(TP) + ' TN:' + str(TN) + ' FP:' + str(FP) + ' FN:' + str(FN))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * TP / (TP + TP + FP + FN)
    print('accuracy:' + str(accuracy) + '\nprecision:' + str(precision) + '\nrecall:' + str(recall) + '\nF1:' + str(F1))
    with open(PathRoot + 'log/' + dataName+ '/'+ netName + '/测试结果.txt', 'a') as file_handle:
        file_handle.write('测试时间为:' + str(testTime))
        file_handle.write('\nT-E-hidden_size:' + str(T_E_hidden) + '  T-E-drop_out:' + str(T_E_dropout)+' T_E_dropout'+str(T_E_dropout))
        file_handle.write('\nTP:' + str(TP) + ' TN:' + str(TN) + ' FP:' + str(FP) + ' FN:' + str(FN))
        file_handle.write('\naccuracy:' + str(accuracy) + '\nprecision:' + str(precision) + '\nrecall:' + str(recall) + '\nF1:' + str(F1))