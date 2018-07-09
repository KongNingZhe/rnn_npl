import tensorflow as tf
import rnn
import dense

n_input = 300
n_steps = 40
n_hidden = 256
hidden_sizes = [512,512,256,256,128,128,64,64,20,20]

global_steps = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.0001, global_steps,1000,0.9, staircase=False)

def placeholder():
    x1 = tf.placeholder(tf.float32, [None, n_steps, n_input])
    x2 = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None])
    dropout1 = tf.placeholder(tf.float32, shape=())
    dropout5 = tf.placeholder(tf.float32, shape=())
    return x1,x2,y,dropout1,dropout5
def model(x1,x2,dropout1,dropout5):
    rnn_x1 = rnn.BiRNN(x1,n_hidden,dropout1,dropout5)
    feature_x1 = dense.Dense(rnn_x1,hidden_sizes,dropout5)
    rnn_x2 = rnn.BiRNN(x2,n_hidden,dropout1,dropout5)
    feature_x2 = dense.Dense(rnn_x2,hidden_sizes,dropout5)
    print ("have built dense")
    l2_x1 = tf.nn.l2_normalize(feature_x1, dim=1)
    l2_x2 = tf.nn.l2_normalize(feature_x2, dim=1)
    #pred = tf.losses.cosine_distance(l2_x1,l2_x2,reduction=tf.losses.Reduction.NONE, dim=1)
    #pred = tf.reduce_sum(tf.multiply(l2_x1,l2_x2), axis = 1)

    #pred = tf.exp(-tf.norm(l2_x1-l2_x2,axis = 1))
    pred = tf.exp(-tf.norm(feature_x1-feature_x2,ord = 1,axis = 1))

    pred = tf.reshape(pred,[-1])
    return pred
def cost(y,pred):
    y_target = tf.cast(y, tf.float32)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    R = tf.reduce_sum(reg_variables) 
    cost = tf.losses.log_loss(labels = y_target,predictions =  pred,epsilon=1e-15)
    loss = cost + R
    print ("have defined the loss and cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_steps)
    return optimizer,cost
def model2(x):
    rnn_x1 = rnn.BiRNN(x,n_hidden,1.0,1.0)
    print ("have built rnn")
    feature_x1 = dense.Dense(rnn_x1,hidden_sizes,1.0)
    print ("have built dense")
    l2_x1 = tf.nn.l2_normalize(feature_x1, dim=1)
    return l2_x1