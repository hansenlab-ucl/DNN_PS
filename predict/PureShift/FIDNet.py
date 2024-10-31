import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as     tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#
# define my own activations to help the gradient optimizer
def dfh_tanh(x):
    return tf.math.tanh(x)*tf.constant(0.98, dtype=x.dtype) + x*tf.constant(0.02,dtype=x.dtype)
def dfh_sigmoid(x):
    return tf.math.sigmoid(x)*tf.constant(0.98, dtype=x.dtype) + x*tf.constant(0.02,dtype=x.dtype)
def dfh_relu(x):
    return tf.nn.relu(x)*tf.constant(0.98, dtype=x.dtype) + x*tf.constant(0.02,dtype=x.dtype)

def hilbert(x):
    """
    Compute the analytic signal, using the Hilbert transform in tensorflow.
    The transformation is done along the last axis by default.

    Flemming, March 2022

    """
    if x.dtype.is_complex:
        raise ValueError("x must be real.")
    #
    N = x.get_shape()[-1]
    #
    # Do forward fft
    Xf = tf.signal.fft(tf.cast(x,dtype=tf.complex64))
    #
    # Make unit-step function vector 
    hh = tf.concat( [tf.constant([1.], dtype=x.dtype),
                     2*tf.ones(N//2-1, dtype=x.dtype),
                     tf.constant([1.], dtype=x.dtype),
                     tf.zeros(N//2-1,  dtype=x.dtype),
                    ], axis=-1)
    hh = tf.complex( hh, 0. )
    X_conv = tf.math.multiply(Xf,hh)
    #
    # inverse fft
    X_ifft = tf.signal.ifft(X_conv)
    return 1.0*X_ifft


class FIDNetLayer(tf.keras.layers.Layer):
  def __init__(self, filters=32, kernel=(3,8), blocks=1, dilations=[1,2,3,4,6,8,10,12,14,16,18,20,24,28,32]):
    super(FIDNetLayer, self).__init__()
    self.filters=filters
    self.kernel=kernel
    self.blocks=blocks
    self.dilations=dilations

    self.conv_y1=[]
    self.conv_y2=[]
    self.conv_z0=[]
    self.dense_z=[]

    #define layers
    for b in range(self.blocks):
        for i in range(len(self.dilations)):
            dil = self.dilations[i]
            self.conv_y1.append( tf.keras.layers.Conv2D(filters=self.filters,   kernel_size=self.kernel, padding='valid', dilation_rate=[1,dil] ))
            self.conv_y2.append( tf.keras.layers.Conv2D(filters=self.filters,   kernel_size=self.kernel, padding='valid', dilation_rate=[1,dil] ))
            self.conv_z0.append( tf.keras.layers.Conv2D(filters=self.filters*2, kernel_size=self.kernel, padding='valid', dilation_rate=[1,1]   ))
            #self.dense_z.append( tf.keras.layers.Dense(4,use_bias=False) )

    self.init_dense = tf.keras.layers.Dense(2*self.filters, activation=None) #tanh


  def waveLayer(self, x, counter):

    dil = self.dilations[counter % len(self.dilations) ]

    xin=tf.pad( x, [ [0,0], [(self.kernel[0]-1)//2,(self.kernel[0]-1)//2], [0, dil*(self.kernel[1]-1)],[0,0]] , "CONSTANT", 0.)
    
    y1 = self.conv_y1[counter](xin)
    y2 = self.conv_y2[counter](xin)
    
    y1 = dfh_tanh(y1)
    y2 = dfh_sigmoid(y2)

    z = y1*y2
    z=tf.pad( z, [ [0,0], [(self.kernel[0]-1)//2,(self.kernel[0]-1)//2], [0, (self.kernel[1]-1)],[0,0]] , "CONSTANT", 0.)
    z = self.conv_z0[counter](z)

    return z

  def call(self, x):

    x = self.init_dense(x)
    x = dfh_tanh(x)
    #
    skips=[]
    for b in range(self.blocks):
        for dd in range(len(self.dilations)):
            xw=self.waveLayer(x,dd + len(self.dilations)*b )
            skips.append(xw)
            x = xw + x

    x = tf.math.reduce_sum(tf.stack( skips, axis=-1), axis=-1)
    x = dfh_tanh(x)
    
    return x  

class OneDFIDNet(tf.keras.Model):
  def __init__(self, fidnet_filters=32, \
               blocks=3, \
               fidnet_kernel=(3,8), \
               dilations=[1,2,3,4,6,8,10,12,14,16,18,20,24,28,32], \
               refine_kernel=(1,8), \
               refine_steps=4, \
               rate=None ):
    super().__init__()

    self.fidnet_filters = fidnet_filters
    self.refine_steps=refine_steps
    self.rate = rate
    self.dilations = dilations
    
    self.fidnet = FIDNetLayer(filters=fidnet_filters,  \
                              blocks=blocks,           \
                              kernel=fidnet_kernel,    \
                              dilations=dilations      \
    )
    #
    # Post FIDNet conv
    self.postfidnet0 = tf.keras.layers.Conv2D(filters=fidnet_filters*2, kernel_size=fidnet_kernel, padding="same")
    self.postfidnet1 = tf.keras.layers.Conv2D(filters=fidnet_filters*2, kernel_size=fidnet_kernel, padding="same")
    
    #
    # final conv2D layers
    self.conv_ref_r=[]
    self.conv_ref_t=[]
    for i in range(self.refine_steps):
        self.conv_ref_r.append( tf.keras.layers.Conv2D(filters=fidnet_filters*2 + 5, kernel_size=refine_kernel, padding='same', dilation_rate=[1,1] ))
        self.conv_ref_t.append( tf.keras.layers.Conv2D(filters=fidnet_filters*2 + 5, kernel_size=refine_kernel, padding='same', dilation_rate=[1,1] ))
    self.dense   = tf.keras.layers.Dense( 2, use_bias=True, activation=None)

    if rate is not None:
        self.dropout0 = tf.keras.layers.Dropout(rate)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, inputs, training=False):
    times, inp, MyScaling = inputs

    inp     = tf.transpose( inp,   perm=(0,2,1)) 
    times   = tf.transpose( times, perm=(0,2,1)) #

    inp = tf.expand_dims( inp, axis=1)   
    # 
    cout = self.fidnet(inp) 
    cout = self.postfidnet0(cout) 
    cout = dfh_relu(cout) 
    cout = self.postfidnet1(cout) 
    cout = dfh_tanh(cout)     
    
    if self.rate is not None:
        cout = self.dropout0(cout, training=training)
 
    cout = tf.cast( cout, tf.float32)
    cout = tf.transpose( cout, perm=(0,3,1,2) )
    #
    # Zero fill
    cout = tf.pad( cout, [ [0,0],[0,0],[0,0],[0,cout.shape[-1]]], "CONSTANT", constant_values=0.)
    multiplier = tf.concat( [ tf.constant([0.5,], dtype=cout.dtype), tf.ones( (cout.shape[-1]//2-1,), dtype=cout.dtype)], axis=0)
    multiplier = tf.complex( tf.reshape( multiplier, (1,1,-1)), tf.constant(0.,dtype=multiplier.dtype))
    #
    cout_ft= tf.signal.fftshift( tf.signal.fft( \
                                                multiplier*tf.complex(cout[:,:,:,0::2], cout[:,:,:,1::2]) \
    ) , axes=-1)
    cout_ft = tf.math.real(cout_ft)
    cout_ft = tf.transpose(cout_ft, perm=(0,2,3,1)) 
    cout_ft = tf.cast( cout_ft, inp.dtype)    
    #
    # FT the input
    inp_ft = tf.cast( inp, tf.float32)
    inp_ft = tf.transpose(inp_ft, perm=(0,3,1,2)) 
    inp_ft = tf.pad( inp_ft, [ [0,0],[0,0],[0,0],[0,inp_ft.shape[-1]]], "CONSTANT", constant_values=0.)
    inp_ft= tf.signal.fftshift( tf.signal.fft( \
                                                multiplier*tf.complex(inp_ft[:,:,:,0::2], inp_ft[:,:,:,1::2]) \
    ) , axes=-1)
    inp_ft = tf.math.real(inp_ft)
    inp_ft = tf.transpose(inp_ft, perm=(0,2,3,1)) 
    inp_ft = tf.cast( inp_ft, inp.dtype)
    #

    hout_ft = tf.concat([cout_ft, MyScaling*inp_ft], axis=-1) 
    #
    ref = hout_ft

    fnorm = tf.reduce_max( tf.abs(ref), axis=(1,2,3), keepdims=True )
    ref = ref / fnorm
    ref = tf.tile( ref, (1,1,1,2))
    #
    for i in range(self.refine_steps):
        rr = dfh_relu(self.conv_ref_r[i](ref))
        tt = dfh_tanh(self.conv_ref_t[i](ref))
        ref = tf.concat([tt,rr], axis=-1) + ref
    ref *= fnorm

    hout_final = tf.concat([ MyScaling*ref, (2.-MyScaling)*hout_ft], axis=-1) #
    hout_final = self.dense(hout_final)
    #
    hout_final = tf.stack( [5.*hout_final[...,0], hout_final[...,1]/fnorm[...,0]], axis=-1)

    hout_final = tf.squeeze( hout_final, axis=1)
    hout_final = tf.cast( hout_final, tf.float32)

    return hout_final
    
