import pandas as pd
import tensorflow as tf
import numpy as np
import os
import model 
import database as db
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#pu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto()
config.gpu_options.allow_growth =True


MODEL = 'test'
MODEL_PATH = './' + MODEL + '/model.ckpt'

TEST_PATH = '../data_set/test.csv'
QUESTION_PATH = '../data_set/question.csv'
W_EMBED_PATH = '../data_set/word_embed.txt'

print ('Load files...')
questions = pd.read_csv(QUESTION_PATH)
test = pd.read_csv(TEST_PATH)
w_embed = db.read_embed(W_EMBED_PATH)

x1_data,x2_data = db.database_test(test,questions,w_embed)

n_input = 300
batch_size = 1
n_step

x1 = tf.placeholder(tf.float32, [ None,n_step ,n_input])
x2 = tf.placeholder(tf.float32, [ None,n_step, n_input])

dropout1 = tf.placeholder(tf.float32,shape=())
dropout5 = tf.placeholder(tf.float32,shape=())

pred = model.model(x1,x2,dropout1,dropout5)

saver = tf.train.Saver()
with tf.Session(config =config) as sess:
    #ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess,MODEL_PATH)
    print ("for money")
    pre = []
    f = 100
    b = len(x1_data) // f
    for i in range(f+1):
        predict_prob = sess.run(pred, feed_dict={x1:x1_data[i*b:i*b+b],x2:x2_data[i*b:i*b+b],dropout1:1.0,dropout5:1.0})
        predict_prob = list(predict_prob)
        pre = pre + predict_prob
        #print (predict_prob)
    with open('submission.csv', 'w') as file:
          file.write(str('y_pre') + '\n')
          for line in pre:
              file.write(str(line) + '\n')
    file.close()
