import pandas as pd
import numpy as np
import random

TRAIN_PATH = '../data_set/train.csv'
TEST_PATH = '../data_set/test1.csv'
MONEY_PATH = '../data_set/test.csv'
QUESTION_PATH = '../data_set/question.csv'
W_EMBED_PATH = '../data_set/word_embed.txt'

n_steps = 40

def read_embed(embed_path):
    f = open(embed_path,'r')
    embed = f.readlines()
    w_embed=dict()
    for i in embed:
        ebed = i.strip().split(' ')
        w_embed[ebed[0]] = list(map(float,ebed[1:301]))
    return w_embed

def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)

def get_texts(file_path, question_path):
    qes = pd.read_csv(question_path)
    file = pd.read_csv(file_path)
    q1id, q2id = file['q1'], file['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = qes['words']
    texts = []
    for t_ in zip(id1s, id2s):
        texts.append(all_words[t_[0]] + ' ' + all_words[t_[1]])
    return texts

def database(f,q,e,maxlen):
    q1id,q2id = f['q1'],f['q2']
    label = f['label']
    ox1 = []
    ox2 = []
    labels = []
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = q['words']
    # k = [0] * 300
    # x = [k] * maxlen
    i = 0
    for t_ in zip(id1s, id2s):
        x1_data= [[0]*300]*maxlen
        x2_data= [[0]*300]*maxlen
        x1,x2 = all_words[t_[0]].strip().split(' '),all_words[t_[1]].strip().split(' ')
        if len(x1) >20 or len(x2) > 20:
            i = i+1
            continue
        for index,x_ in enumerate(x1):
            x1_data[index] = e[x_]
        ox1.append(x1_data)
        for index,x_ in enumerate(x2):
            x2_data[index] = e[x_]
            #print(index)
        ox2.append(x2_data)
        labels.append(label[i])
        i = i+1
    print ("数据大小",len(ox1),"l",len(labels))
    return ox1,ox2,labels

def database_test(f,q,e):
    q1id,q2id = f['q1'],f['q2']
    ox1=[]
    ox2=[]
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = q['words']
    k = [0] * 300
    x = [k] * 39
    for t_ in zip(id1s, id2s):
         x1,x2 = all_words[t_[0]].strip().split(' '),all_words[t_[1]].strip().split(' ')
         l1,l2 = len(x1),len(x2)
         x1_data= [[0]*300]*20
         x2_data= [[0]*300]*20
         for index,x_ in enumerate(x1):
             if l1 < 21 :
                x1_data[index] = e[x_]
         ox1.append(x1_data)
         for index,x_ in enumerate(x2):
             if l2 < 21:
                x2_data[index] = e[x_]
             #print(index)
         ox2.append(x2_data)
    return ox1,ox2

def get_batch(batch_size,x1,x2,label):
     l=len(x1)
     #l2= len(label)
     #print(l)
     #print(l2)
     r = random.sample([i for i in range(l)],batch_size)
     b_x1 = [x1[i] for i in r]
     b_x2 = [x2[i] for i in r]
     b_label = [label[i] for i in r]
     return b_x1,b_x2,b_label
def short_for_data():
    print('Load files...')
    questions = pd.read_csv(QUESTION_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    m_test = pd.read_csv(MONEY_PATH)
    w_embed = read_embed(W_EMBED_PATH)
    x1_data,x2_data,label = database(train,questions,w_embed,n_steps)
    Tx1_data,Tx2_data,Ty_data = database(test,questions,w_embed,n_steps)

    #m_x1,m_x2 = database_test(m_test,,w_embed)
    print("data over")
    return x1_data,x2_data,label,Tx1_data,Tx2_data,Ty_data