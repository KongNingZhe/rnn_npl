import tensorflow as tf

__REGULAR__ = 0.0000001

regular = tf.keras.regularizers.l2(l = __REGULAR__)
initializer = tf.contrib.layers.xavier_initializer()

def Layer(x,hidden_size,i,drop_remain):
    nn = tf.layers.dense(
                        x,
                        hidden_size,
                        activation = tf.nn.selu,
                        kernel_regularizer = regular,
                        kernel_initializer = initializer,
                        name = 'dense'+str(i),
                        reuse = tf.AUTO_REUSE
                        )
    out = tf.layers.dropout(nn,rate = drop_remain,name= 'dropout'+str(i))
    return out

def LayerS(x,hidden_size,i,drop_remain):
    nn = tf.layers.dense(
                        x,
                        hidden_size,
                        activation = tf.nn.softmax,
                        kernel_regularizer = regular,
                        kernel_initializer = initializer,
                        name = 'dense'+str(i),
                        reuse = tf.AUTO_REUSE
                        )
    out = tf.layers.dropout(nn,rate = drop_remain,name= 'dropout'+str(i))
    return out

def Dense(input,hidden_sizes,drop_remain):
    i = 1
    for h in hidden_sizes:
        out  = Layer(input,h,i,drop_remain)
        input = out
        i = i+1
    #out = LayerS(input,hidden_sizes[-1],i,drop_remain)
    #print (nn3_d,'nn3_d')
    return out

#############人生啊#################
# def Dense(input,hidden_sizes,drop_remain):
#     last_word = input
#     nn1 = tf.layers.dense(last_word,512,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn1_d = tf.layers.dropout(nn1,rate = drop_remain)
#     nn2 = tf.layers.dense(nn1_d,512,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib    .layers.xavier_initializer())
#     nn2_d = tf.layers.dropout(nn2,rate = drop_remain)
#     nn3 = tf.layers.dense(nn2_d,256,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn3_d = tf.layers.dropout(nn3,rate = drop_remain)
#     nn4 = tf.layers.dense(nn3_d,256,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn4_d = tf.layers.dropout(nn4,rate = drop_remain)
#     nn5 = tf.layers.dense(nn4_d,128,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn5_d = tf.layers.dropout(nn5,rate = drop_remain)
#     nn6 = tf.layers.dense(nn5_d,128,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn6_d = tf.layers.dropout(nn6,rate = drop_remain)
#     nn7 = tf.layers.dense(nn6_d,64,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn7_d = tf.layers.dropout(nn7,rate = drop_remain)
#     nn8 = tf.layers.dense(nn7_d,64,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn8_d = tf.layers.dropout(nn8,rate = drop_remain)
#     nn9 = tf.layers.dense(nn8_d,20,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn9_d = tf.layers.dropout(nn9,rate = drop_remain)
#     nn10 = tf.layers.dense(nn9_d,20,activation = tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l=0.01),kernel_initializer = tf.contrib.layers.xavier_initializer())
#     nn10_d = tf.layers.dropout(nn10,rate = drop_remain)
#     #print (nn3_d,'nn3_d')
#     return nn10_d