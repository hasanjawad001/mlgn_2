#
# XAenc implementation
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb
#

import tensorflow as tf
import numpy as np

#--------------------------------------------------------------------------
# scaled_dot_product_attention
#-------------------------------------------------------------------------- 
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  attention_weights *= (1-mask)

#   print('atten_weights')
#   print(attention_weights)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

#   print('output')
#   print(output)

  return output, attention_weights

#--------------------------------------------------------------------------
# point_wise_feed_forward_network
#-------------------------------------------------------------------------- 
def point_wise_feed_forward_network(d_model, dff, rate):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', use_bias=False),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dropout(rate),
      tf.keras.layers.Dense(d_model, use_bias=False)  # (batch_size, seq_len, d_model)
  ])

#--------------------------------------------------------------------------
# invert_feed_forward_network
#-------------------------------------------------------------------------- 
def invert_feed_forward_network(dff, rate):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', use_bias=False),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dropout(rate),
      tf.keras.layers.Dense(dff, activation='relu', use_bias=False),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dropout(rate),
      tf.keras.layers.Dense(1, use_bias=False)  # (batch_size, seq_len, 1)
  ])


#--------------------------------------------------------------------------
# encoder_network
#-------------------------------------------------------------------------- 
def encoder_network(dff, d_model, rate):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dropout(rate),
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dropout(rate),
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


#--------------------------------------------------------------------------
# SubNetwork
#-------------------------------------------------------------------------- 
class SubNetwork(tf.keras.layers.Layer):
  def __init__(self, architecture, rate):
    super(SubNetwork, self).__init__()

    self.architecture = architecture
    #self.sn_layers = [tf.keras.layers.Dense(N) for N in architecture]
    
    self.sn_layers = []
    for i in range(len(architecture)):
        N = architecture[i]
        if i < len(architecture)-1:
            self.sn_layers.append( tf.keras.layers.Dense(N, activation='relu') )
            self.sn_layers.append( tf.keras.layers.Dropout(rate) )
        else:
            self.sn_layers.append( tf.keras.layers.Dense(N) )
            
  def call(self, x):
    
    for my_layer in self.sn_layers:
        x = my_layer(x)
    
    return x

#--------------------------------------------------------------------------
# MultiHeadAttention
#-------------------------------------------------------------------------- 
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, architecture, num_heads):
    super(MultiHeadAttention, self).__init__()
    
    self.num_heads = num_heads
    
    d_model = architecture[-1]
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

#     self.wq = SubNetwork(architecture)
#     self.wk = SubNetwork(architecture)
#     self.wv = SubNetwork(architecture)
    
    self.dense = tf.keras.layers.Dense(d_model, use_bias=False)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

#     print('concat_attention')
#     print(concat_attention)    
    
    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    
#     print('output')
#     print(output)     
    
    return output, attention_weights

#--------------------------------------------------------------------------
# EncoderLayer
#-------------------------------------------------------------------------- 
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, architecture, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    
    d_model = architecture[-1]

    self.mha = MultiHeadAttention(architecture, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff, rate)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    out1 *= (1-tf.transpose(tf.reduce_prod(mask, axis=-1), perm=[0, 2, 1]))
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    out2 *= (1-tf.transpose(tf.reduce_prod(mask, axis=-1), perm=[0, 2, 1]))
    
#     print('out2')
#     print(out2)
    
    return out2, attn_weights

#--------------------------------------------------------------------------
# Encoder
#-------------------------------------------------------------------------- 
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, architecture, num_heads, dff, rate=0.1):
    super(Encoder, self).__init__()

    d_model = architecture[-1]
    
    self.d_model = d_model
    self.num_layers = num_layers
    
    self.enc_layers = [EncoderLayer(architecture, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)

  def build(self, input_shape):
    
    # no attributes
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    #self.embedding = tf.keras.layers.Embedding(seq_len, self.d_model)
    self.embedding = tf.keras.layers.Embedding(seq_len, self.d_model-1)

  def call(self, x, training, mask):

    # no attributes
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]

    # create initial embeddings
    ind = tf.range(seq_len)
    #tmp_embd = self.embedding(ind)*tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    tmp_embd = self.embedding(ind)*tf.math.sqrt(tf.cast(self.d_model-1, tf.float32))
    tmp_embd = tf.expand_dims(tmp_embd,0)
    attr_embd = tf.tile(tmp_embd,(batch_size,1,1))
    attr_embd *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    
    # adding embedding initial embedding
    x = tf.expand_dims(x,2)
    #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    #x += attr_embd
    x = tf.concat([attr_embd, x], axis=2)
    x = self.dropout(x, training=training)
    
    x *= (1-tf.transpose(tf.reduce_prod(mask, axis=-1), perm=[0, 2, 1]))
#     print('x')
#     print(x)

    attn_weights = {}
    for i in range(self.num_layers):
      x, attn_weights[i] = self.enc_layers[i](x, training, mask)
    
    return x, attn_weights  # (batch_size, input_seq_len, d_model)

#--------------------------------------------------------------------------
# CustomSchedule
#-------------------------------------------------------------------------- 
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# #--------------------------------------------------------------------------
# # XAModel
# #-------------------------------------------------------------------------- 
# class XAModel(tf.keras.Model):

#     def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
#         super(XAModel, self).__init__()
        
#         self.encoder = Encoder(num_layers, d_model, num_heads, dff)
#         self.ffn_att = invert_feed_forward_network(dff, rate)
#         self.ffn = point_wise_feed_forward_network(d_model, dff)
#         self.bias = tf.Variable(initial_value=tf.random.normal((1,1)), trainable=True)
        
#     def get_everything(self, inputs, training):

#         batch_size = tf.shape(inputs)[0]
#         seq_len = tf.shape(inputs)[1]

#         # mask is going to be extracted by finding NaNs
#         inputs_nan = tf.math.is_nan(inputs)
#         inputs = tf.where(inputs_nan, tf.zeros_like(inputs), inputs)
#         mask = tf.cast(inputs_nan, tf.float32)[:, tf.newaxis, tf.newaxis, :]
#         #mask = None

#         # get final embeddings
#         tmp_embd = self.encoder(inputs, training, mask) # (batch_size, input_seq_len, d_model)
#         final_embd = self.ffn(tmp_embd)

#         M = tf.matmul(final_embd, final_embd, transpose_b=True)
#         phi = tf.math.reduce_sum(M, axis=2, keepdims=True)
#         y = tf.math.reduce_sum(phi, axis=1, keepdims=False) # (batch_size, 1)
#         y += self.bias

#         y_att = self.ffn_att(final_embd) # (batch_size, input_seq_len, 1)
#         y_att = tf.reshape(y_att, shape=(batch_size, seq_len))
#         y_all = tf.concat([y_att, y], axis = 1)

#         if training:
#             pass
        
#         return y_all, phi, final_embd
        
#     def call(self, inputs, training):
        
#         y, phi, final_embd = self.get_everything(inputs, training)
        
#         return y

# #--------------------------------------------------------------------------
# # GTModel
# #-------------------------------------------------------------------------- 
# class GTModel(tf.keras.Model):

#     def __init__(self, d_model, dff, no_outputs=1, rate=0.1):
#         super(GTModel, self).__init__()

#         self.d_model = d_model
#         self.dff = dff
#         self.no_outputs = no_outputs
#         self.rate = rate

#     def build(self, input_shape):
      
#         # no attributes
#         batch_size = input_shape[0]
#         seq_len = input_shape[1]
#         self.embedding = tf.keras.layers.Embedding(seq_len, self.d_model)
      
#         #self.ffn_att = invert_feed_forward_network(self.dff, self.rate)
#         self.enc_nn = encoder_network(self.dff, self.d_model, self.rate)
#         self.bias = tf.Variable(initial_value=tf.random.normal((1,1)), trainable=True)
        
#     def get_everything(self, inputs, training):

#         batch_size = tf.shape(inputs)[0]
#         seq_len = tf.shape(inputs)[1]

#         inputs = tf.cast(inputs, tf.float32)

#         # create initial embeddings
#         ind = tf.range(seq_len)
#         tmp_embd = self.embedding(ind)*tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#         #tmp_embd = self.embedding(ind)*tf.math.sqrt(tf.cast(self.d_model-1, tf.float32))
#         tmp_embd = tf.expand_dims(tmp_embd,0)
#         attr_embd = tf.tile(tmp_embd,(batch_size,1,1))
#         attr_embd *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
#         # adding embedding initial embedding
#         inputs = tf.expand_dims(inputs,2)
#         inputs *= attr_embd
#         #inputs = tf.concat([attr_embd, inputs], axis=2)

#         # TODO - incorporate the mask
#         # mask is going to be extracted by finding NaNs
#         # inputs_nan = tf.math.is_nan(inputs)
#         # inputs = tf.where(inputs_nan, tf.zeros_like(inputs), inputs)
#         # mask = tf.cast(inputs_nan, tf.float32)[:, tf.newaxis, tf.newaxis, :]
#         mask = None

#         # get final embeddings
#         final_embd = self.enc_nn(inputs)

#         M = tf.matmul(final_embd, final_embd, transpose_b=True)
#         phi = tf.math.reduce_sum(M, axis=2, keepdims=True)
#         y = tf.math.reduce_sum(phi, axis=1, keepdims=False) # (batch_size, 1)
#         y += self.bias

#         # y_att = self.ffn_att(final_embd) # (batch_size, input_seq_len, 1)
#         # y_att = tf.reshape(y_att, shape=(batch_size, seq_len))
#         # y_all = tf.concat([y_att, y], axis = 1)

#         if training:
#             pass
        
#         return y, phi, final_embd
        
#     def call(self, inputs, training):
        
#         y, phi, final_embd = self.get_everything(inputs, training)
        
#         return y


#--------------------------------------------------------------------------
# XAModel
#-------------------------------------------------------------------------- 
class XAModel(tf.keras.Model):

    def __init__(self, num_layers, arch_enc, num_heads, 
                 dff, arch_contrib, type_problem, rate=0.1):
        
        super(XAModel, self).__init__()
        
        d_model = arch_enc[-1]
        no_outputs = arch_contrib[-1]
        
        self.type_problem = type_problem
        self.encoder = Encoder(num_layers, arch_enc, num_heads, dff)
        self.ffn_att = SubNetwork(arch_contrib, rate)
        self.bias = tf.Variable(initial_value=tf.random.normal((1,no_outputs)), trainable=True)
        
        if type_problem == 'classification':
            self.last_layer = tf.keras.layers.Softmax()
        
        #self.ffn_att = invert_feed_forward_network(dff, rate)
        #self.ffn = point_wise_feed_forward_network(d_model, dff, rate)
        #self.activity_reg = tf.keras.layers.ActivityRegularization(l1=rate)
        
    def get_everything(self, inputs, training):

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # mask is going to be extracted by finding NaNs
        inputs_nan = tf.math.is_nan(inputs)
        inputs = tf.where(inputs_nan, tf.zeros_like(inputs), inputs)
        #mask = tf.cast(inputs_nan, tf.float32)[:, tf.newaxis, tf.newaxis, :]
        mask = tf.expand_dims(inputs_nan,2) | tf.expand_dims(inputs_nan,1) 
        mask = tf.cast(mask, tf.float32)[:, tf.newaxis, :, :]
        #print('Mask')
        #print(mask)
        #mask = None

        # get final embeddings
        final_embd, attn_weights = self.encoder(inputs, training, mask) # (batch_size, input_seq_len, d_model)
        #final_embd = self.ffn(tmp_embd)

        phi = self.ffn_att(final_embd) # (batch_size, input_seq_len, no_outputs)
        y = tf.math.reduce_sum(phi, axis=1, keepdims=False) # (batch_size, no_outputs)
        y += self.bias # (batch_size, no_outputs)
        
        if self.type_problem == 'classification':
            y_logits = y
            y = self.last_layer( y )

        if training:
            pass
        
        # add activity regularization for the final embbedings
#         innerProd = tf.matmul( final_embd, tf.transpose(final_embd, perm=(0,2,1)) )
#         avgAbsInnerProd = tf.reduce_mean( tf.abs( innerProd ), axis=0 )
#         sM = tf.expand_dims( tf.sqrt(tf.reduce_sum(final_embd*final_embd, axis=2)), axis=2)
#         norm = tf.matmul(sM, tf.transpose(sM, perm=(0,2,1)))
#         simMat = tf.reduce_mean( tf.abs(innerProd/norm), axis=0 )
        
        #simMat = self.activity_reg(avgAbsInnerProd)
#         simMat = tf.reduce_mean( innerProd*innerProd, axis=0 )
#         simMat = self.activity_reg( simMat )
        
#         innerProd = self.activity_reg( innerProd )
        
        #attn_weights = self.activity_reg(attn_weights)
    
        if self.type_problem == 'classification':
            return y, y_logits, phi, final_embd, attn_weights
        else:
            return y, phi, final_embd, attn_weights
        
    def call(self, inputs, training):
        
        if self.type_problem == 'classification':
            y, y_logits, phi, final_embd, attn_weights = self.get_everything(inputs, training)
        else:
            y, phi, final_embd, attn_weights = self.get_everything(inputs, training)
        
        return y