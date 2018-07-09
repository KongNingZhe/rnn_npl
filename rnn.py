import tensorflow as tf

__MAX_LEN__ = 40

def LSTMCELL(n_hidden,in_keep,out_keep):
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0,reuse=tf.AUTO_REUSE)
    out = tf.contrib.rnn.DropoutWrapper(cell,
                                        input_keep_prob=in_keep,
                                        output_keep_prob=out_keep)
    return out
def trans(x):
    x = tf.transpose(x, [1, 0, 2])
    n_embed = int(x.get_shape()[2])
    x = tf.reshape(x, [-1, n_embed])
    out = tf.split(x,__MAX_LEN__)
    return out
def BiRNN(x,n_hidden,drop1,drop5):
    x = trans(x)
    print ("trans the x")
    with  tf.variable_scope('Birnn') as scope:
        f_cellA = LSTMCELL(n_hidden,drop1,drop1)
        #print (f_cellA)
        f_cellm1 = LSTMCELL(n_hidden,drop5,drop5)
        #print (f_cellm1)
        f_cellm2 = LSTMCELL(n_hidden,drop5,drop5)
        f_cellm3 = LSTMCELL(n_hidden,drop5,drop5)
        f_cellm4 = LSTMCELL(n_hidden,drop5,drop5)
        f_cellZ = LSTMCELL(n_hidden,drop1,drop1)

        LSTM_fw_cells = [f_cellA,f_cellm1,f_cellm2,f_cellm3,f_cellm4,f_cellZ]
        #print (LSTM_fw_cells)
        b_cellA = LSTMCELL(n_hidden,drop1,drop1)
        b_cellm1 = LSTMCELL(n_hidden,drop5,drop5)
        b_cellm2 = LSTMCELL(n_hidden,drop5,drop5)
        b_cellm3 = LSTMCELL(n_hidden,drop5,drop5)
        b_cellm4 = LSTMCELL(n_hidden,drop5,drop5)
        b_cellZ = LSTMCELL(n_hidden,drop1,drop1)
        
        LSTM_bw_cells = [b_cellA,b_cellm1,b_cellm2,b_cellm3,b_cellm4,b_cellZ]
        #print (LSTM_bw_cells)
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(
            LSTM_fw_cells,
            LSTM_bw_cells,
            x,
            dtype=tf.float32
        )
    return outputs[-1]

def lstm_cell(size):
  return tf.contrib.rnn.BasicLSTMCell(
      size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)