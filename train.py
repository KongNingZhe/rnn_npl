import pandas as pd
import copy
import random
import tensorflow as tf
import numpy as np
import os

import database as db
import rnn
import dense
import model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth =True

learning_rate = 0.0001
max_samples = 254386 * 10
a_sample = 254386
batch_size = 256
display_step = 10

n_input = 300
n_steps = 20
n_hidden = 256
hidden_sizes = [512,512,256,256,128,128,64,64,20,20]

ruler = 0.15

x1,x2,y,Tx1,Tx2,Ty = db.short_for_data()
print("data over")
x1p,x2p,yp,dropout1,dropout5 = model.placeholder()
print ("have the placeholder")
pre = model.model(x1p,x2p,dropout1,dropout5)
print ("have got the pred")
opt,cost = model.cost(yp,pre)
print ("have the optimizer")

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=0)
with tf.Session(config =config) as sess:
    print ("in the session")
    sess.run(init)
    print ("init over")
    step = 1
    t_losses = 10
    while ruler < t_losses:
        batch_x1, batch_x2,batch_y = db.get_batch(batch_size,x1,x2,y)
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(opt, feed_dict={x1p: batch_x1,x2p:batch_x2,yp:batch_y,dropout1:1.0,dropout5:0.5})
        #print ("one circle")
        if step % display_step == 0:
            #acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            p,losses = sess.run([pre,cost], feed_dict={x1p: batch_x1,x2p:batch_x2,yp:batch_y,dropout1:1.0,dropout5:1.0})
            losses2 = sess.run(cost, feed_dict={x1p: batch_x1,x2p:batch_x2,yp:batch_y,dropout1:1.0,dropout5:0.5})
            print ("Iter" + str(step * batch_size) + ", Minibatch Loss=" + \
                   "{:.6f}".format(losses)+"   {:.6f}".format(losses2))
            print("predloook:"+str(p[0:4]))
            # if losses < 0.22:
            #    path = './model-' + 'trainloss-' + str(losses) + '-' + str(step)
            #    os.mkdir(path)
            #    print ("模型已保存")
            #    path = path + '/model.ckpt'
            #    saver.save(sess, path)
        step += 1
        if step * batch_size % a_sample < batch_size:
            t_losses = 0
            f = 100
            b = len(Tx1) // f
            for i in range(f):
                t_losses += sess.run(cost, feed_dict={x1p:Tx1[i*b:i*b+b],x2p:Tx2[i*b:i*b+b],yp:Ty[i*b:i*b+b],dropout1:1.0,dropout5:1.0})
            t_losses = t_losses / f
            if t_losses < 0.22:
              path = './model-' + 'testloss-' + str(t_losses) + '-' + str(step)
              os.mkdir(path)
              print ("模型已保存")
              path = path + '/model.ckpt'
              saver.save(sess, path)
            print ("Testing loss:",t_losses)
    print ("Optimization Finishes!")