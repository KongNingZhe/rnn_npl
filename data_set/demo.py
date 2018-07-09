from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier

from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
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


def make_submission(predict_prob):
    with open('submission.csv', 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            file.write(str(line) + '\n')
    file.close()


TRAIN_PATH = './train1.csv'
TEST_PATH = './test1.csv'
QUESTION_PATH = './question.csv'

print('Load files...')
questions = pd.read_csv(QUESTION_PATH)
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
corpus = questions['words']

print('Fit the corpus...')
vec = TfidfVectorizer()
vec.fit(corpus)

print('Get texts...')
train_texts = get_texts(TRAIN_PATH, QUESTION_PATH)
test_texts = get_texts(TEST_PATH, QUESTION_PATH)

print('Generate tfidf features...')
tfidf_train = vec.transform(train_texts[:])
tfidf_test = vec.transform(test_texts[:])

print('Train classifier...')

#clfs = [RandomForestClassifier(n_estimators=128, n_jobs = -1, max_depth=None,min_samples_split=2, criterion='gini'),
#        RandomForestClassifier(n_estimators=128, n_jobs = -1, max_depth=None,min_samples_split=2, criterion='entropy'),
#       ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
#        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
#        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
#for j, clf in enumerate(clfs):
#    clf.fit(tfidf_train, train['label'][:])
x_train, x_test, y_train, y_test = train_test_split(tfidf_train, train['label'][:], train_size=0.7, random_state=1)
data_train = xgb.DMatrix(x_train,label=y_train)
data_test=xgb.DMatrix(x_test,y_test)
param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 3
watchlist = [ (tfidf_train,'train'), (tfidf_test, 'test') ]
num_round = 10
bst = xgb.train(param, tfidf_train, num_round, watchlist );
pred = bst.predict( tfidf_test);
print('Predict...')
    #pred = clf.predict_proba(tfidf_test)
#make_submission(pred[:, 1])
logloss = log_loss(test['label'][:],pred[:, 1],eps=1e-15)
print("损失值：",logloss)
print('Complete')
