#!/usr/bin/env python3

import os,sys

Eval = False

if len(sys.argv)>1:
  if sys.argv[1].lower()=='eval':
    Eval=True
  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
if Eval:
  os.environ['CUDA_VISIBLE_DEVICES']=""
    
import tensorflow as     tf
import tensorflow.keras.mixed_precision as mixed_precision
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import copy
import numpy as np 
import matplotlib.pyplot as plt
#import time
import logging
if Eval:
  from MakeTraining_tf_v2_2 import *
else:
  from MakeTraining_tf_v2_2 import *  
from FIDNet import *

#(time, inp), tar = MakeTraining(4)
#print( time.shape, inp.shape, tar.shape )
#sys.exit(10)

#
#Suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

powers = tf.constant([0.5,2.0,3.0,3.5], dtype=tf.complex128)
normal_momenta = tf.random.normal(mean=0.0, stddev=1.0, shape=(16*400*512,1))
normal_momenta = tf.cast( normal_momenta, tf.complex128 )
normal_momenta = tf.math.reduce_mean(tf.pow(normal_momenta, tf.reshape(powers,(1,-1))), axis=0)

StatusFile = sys.argv[0].replace('py','status')
#
# Good hyper parameters
if Eval:
  ROUNDS     =  1
else:
  ROUNDS     = 10000
EPOCHS             = 1
BATCHSIZE          = 16 #24
fidnet_filters  = 48
checkpoint_path = "./checkpointer/fidnet_model_7pts/"

#strategy = tf.distribute.MirroredStrategy()
#strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())

if (not Eval) and False:
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_global_policy(policy)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model=20*20*13, warmup_steps=20000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):  # Step is batch
    step = step
    arg1 = tf.math.rsqrt( tf.cast(step, tf.float32) )
    arg2 = tf.cast(step, tf.float32) * (self.warmup_steps ** -1.5)
    val = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) / 100.
    val = tf.math.maximum( 2.e-7,val)
    return val

with strategy.scope():
  learning_rate = CustomSchedule()

  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  optimizer = mixed_precision.LossScaleOptimizer( optimizer )

  @tf.function
  def loss_function(real, pred):
    vloss = loss_value(real,pred)

    sloss = loss_sigma_3(real,pred)
    tloss = loss_sigma_4(real,pred)*tf.constant(5.,dtype=pred.dtype)
    #
    # we need to ensure no underflow with tf.math.min()
    scaling = tf.math.sigmoid( -tf.math.exp( tf.math.minimum(vloss,0.30) * 10. ))/0.268
    #
    return (2. - scaling )*vloss + 2.*scaling * ( sloss + tloss )
    
  @tf.function
  def loss_sigma_4(real,pred):
    pred_value = pred[...,0]
    pred_sigma = pred[...,1] # sigma
    #
    sigma_list= tf.math.scalar_mul(1.0, pred_sigma )  # (15,400,512)
    diff_list = real - pred_value
    #
    # Reshape 
    sigma_list = tf.reshape( sigma_list, (-1,1))
    diff_list  = tf.reshape( diff_list,  (-1,1))
    #
    sigma_parts = tf.reshape( tfp.stats.quantiles( sigma_list, 200 ), (1,-1))
    #
    # Indices with sigma parts
    idxs = tf.where( ( sigma_parts[:,:-1] < sigma_list ) & \
                     ( sigma_parts[:,1:]  > sigma_list ), tf.constant(1.,dtype=tf.float32) , tf.constant(0.,dtype=tf.float32) ) #(5*400*512,sigma_parts)

    mean_diff_list  = tf.math.reduce_mean(diff_list)
    diff_list_tiles = tf.tile( diff_list, (1,sigma_parts.shape[-1]-1)) # (:, sigma_parts)
    counts = tf.math.reduce_sum( idxs, axis=0, keepdims=True ) #(1,sigma_parts)
    #
    calc_sigma = tf.math.reduce_sum( tf.square( (diff_list_tiles - mean_diff_list) * idxs ), axis=0, keepdims=True)/counts
    calc_sigma = tf.math.sqrt( calc_sigma )
    #
    pred_sigma = tf.math.reduce_sum( sigma_list * idxs, axis=0, keepdims=True )/counts  # (-1,1) * (-1,100) / (1,100)
    #
    #sigma_loss_4 = tf.reduce_mean(tf.square(pred_sigma - calc_sigma))
    #
    # This adds more attention to the low sigmas
    sigma_loss_4 = tf.reduce_mean(tf.math.maximum( \
            tf.square(pred_sigma - calc_sigma),    \
            tf.square(tf.sqrt(pred_sigma) - tf.sqrt(calc_sigma))))
    #
    return sigma_loss_4  

  @tf.function
  def loss_sigma_3(real,pred):
    pred_value = pred[...,0]
    pred_sigma = pred[...,1] # sigma
    #
    sigma_list= tf.math.scalar_mul(1.0, pred_sigma )  # (15,400,512)
    diff_list = real - pred_value
    #
    # Reshape 
    sigma_list = tf.reshape( sigma_list, (-1,1))
    diff_list  = tf.reshape( diff_list,  (-1,1))
    #
    # number of quantiles of sigma.
    # This makes sure that both small and large sigmas will have a chi2
    # distribution that is Gaussian
    sigma_parts = tf.reshape( tfp.stats.quantiles( sigma_list, 20 ), (1,-1))
    #
    # Indices with sigma parts
    idxs = tf.where( ( sigma_parts[:,:-1] < sigma_list ) & \
                     ( sigma_parts[:,1:]  > sigma_list ), tf.constant(1.,dtype=tf.float64) , tf.constant(0.,dtype=tf.float64) ) #(5*400*512,10)
    #idxs = tf.ones(dtype=tf.float64, shape=sigma_list.shape)
    #
    # YAK - we need to work in float64 .. 
    chi2 =      tf.cast(diff_list/sigma_list, tf.float64) # (5*400*512,1)
    #
    chi2 =      tf.tile(chi2, (1,sigma_parts.shape[-1]-1))
    mean_chi2 = tf.math.reduce_mean(chi2)
    #
    counts = tf.math.reduce_sum( idxs, axis=0, keepdims=True ) #(1,10)
    counts = tf.expand_dims( tf.expand_dims( counts, axis=0), axis=-1 )
    idxs   = tf.expand_dims( tf.expand_dims( idxs, axis=0), axis=-1 )
    #
    chi2      = tf.complex( chi2,       tf.constant(0.,dtype=tf.float64) ) #tf.cast( chi2, tf.complex128 )
    mean_chi2 = tf.complex( mean_chi2,  tf.constant(0.,dtype=tf.float64) ) #tf.cast( mean_chi2, tf.complex128)
    idxs      = tf.complex( idxs,       tf.constant(0.,dtype=tf.float64) ) #tf.cast( idxs, tf.complex128 )
    counts    = tf.complex( counts,     tf.constant(0.,dtype=tf.float64) ) #tf.cast( counts, tf.complex128 )
    #
    this_powers = powers
    this_normal_momenta = tf.expand_dims( normal_momenta, axis=0)
    #
    sigma_loss_3c = tf.pow( tf.expand_dims( chi2 - mean_chi2, axis=-1), tf.reshape(this_powers, (1,1,-1)))  # (:, sigma_parts ,powers)
    sigma_loss_3c = tf.expand_dims( sigma_loss_3c, axis=0) # (1,:,sigma,powers)
    sigma_loss_3c = sigma_loss_3c * idxs
    sigma_loss_3c = tf.math.reduce_sum( sigma_loss_3c/counts, axis=(0,1)) #(sigma,powers)
    #
    # this should be more robust - using soft l1 norm
    sigma_loss_3c = tf.abs(this_normal_momenta - sigma_loss_3c)
    sigma_loss_3c = tf.cast( sigma_loss_3c, pred_sigma.dtype )

    sigma_loss_3c_l1 = 2. * (tf.math.sqrt( 1. + sigma_loss_3c) - 1. ) # soft L1
    sigma_loss_3c_l2 = tf.square(sigma_loss_3c)                       # L2

    # soft change between L1 and L2 at ~ 0.5
    scaling = tf.math.sigmoid( ( sigma_loss_3c - 0.5)*10. )
    sigma_loss_3c = sigma_loss_3c_l2*(1. - scaling) + \
      (sigma_loss_3c_l1-0.2)*scaling

    sigma_loss_3c = tf.math.reduce_mean( sigma_loss_3c )

    # we also add a small push on sigma to not be too big    
    return tf.math.scalar_mul(0.1, sigma_loss_3c)
  
  @tf.function
  def loss_value(real,pred):
    pred_val = pred[...,0]
    return tf.math.reduce_mean( tf.math.square( real - pred_val ))

  train_loss = tf.keras.metrics.Mean(name='train_loss')

  #dilations generated as np.int_(np.unique(np.floor(np.power(2, np.power(np.arange(19),0.80)) )))
  fidnet_1d_model = OneDFIDNet(
    fidnet_filters=fidnet_filters,
    blocks=3,
    fidnet_kernel=(1,16),
    refine_kernel=(1,16),
    dilations=[1,2,4,6,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256],
    rate=0.10 )

  ckpt = tf.train.Checkpoint(transformer=fidnet_1d_model,
                             optimizer=optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
  #
  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

    try:
      loss_array=np.load( open(checkpoint_path+'/loss.npy','rb'))
      lr_array = np.load( open(checkpoint_path+'/lr.npy', 'rb'))                        
      loss_array = list(loss_array)
      lr_array = list(lr_array)
    except(FileNotFoundError):
      loss_array=[]
      lr_array=[]
      
    print('\n INFO: Latest checkpoint restored \n')
  else:
    loss_array=[]
    lr_array=[]
    os.system('mkdir -p %s ' %(checkpoint_path))  
  #
  def convolve2(spec, time, transpose=False, offset=0.40, end=0.98, power=2.0):
    spec = hilbert(spec)
    mypi = tf.math.asin(1.)*2.
    #
    spec_time = tf.math.conj(tf.signal.fft( spec ))/tf.complex( tf.cast( spec.shape[-1], tf.float32), 0.)

    # make TD over the batches
    #TD = tf.cast(tf.reduce_max(tf.where( time > 0., tf.range(time.shape[-1], dtype=tf.int32), 0))+1, tf.dtypes.int32)//2
    myrange = tf.range(time.shape[-1], dtype=tf.int32)
    myrange = tf.reshape( myrange, (1,-1))
    myrange = tf.tile( myrange, (time.shape[0],1))
    
    # TD array over batch
    TD = tf.cast(tf.reduce_max(tf.where( time[:,0,...] > 0., myrange, 0), axis=-1)+1, tf.dtypes.int32)//2
    #
    # let's make window.
    if offset is not None:
      myrange = tf.reshape( tf.range(time.shape[-1]//2, dtype=tf.float32), (1,-1))
      window = tf.math.pow(
          tf.math.sin(3.1415*offset + 3.1415*(end-offset)*tf.cast(myrange,tf.float32)/tf.expand_dims(tf.cast(TD,tf.float32),axis=-1))
      ,power)
    else:
      window = tf.ones(shape=(time.shape[0],time.shape[-1]//2), dtype=tf.float32)
    #
    # Zero all larger than TD
    myrange = tf.reshape( tf.range(time.shape[-1]//2, dtype=tf.int32), (1,-1))
    window = tf.where( myrange < tf.reshape(TD, (-1,1)), window, 0.)
    # zero fill
    window = tf.pad( window, [ [0,0], [0,window.shape[-1]]], "CONSTANT", constant_values=0.)
    window = tf.complex( window, 0.)
    #
    spec = tf.signal.fft( spec_time * window )
    return tf.math.real(spec), TD



  def convolve(spec, time, transpose=False, offset=0.40, end=0.98, power=2.0):
    #if transpose:
    #  spec = tf.transpose( spec, perm=(0,2,1))
    spec = hilbert(spec)
    mypi = tf.math.asin(1.)*2.
    #
    TD = tf.cast(tf.reduce_max(tf.where( time > 0., tf.range(time.shape[-1], dtype=tf.int32), 0))+1, tf.dtypes.int32)//2
    spec_time = tf.math.conj(tf.signal.fft( spec ))/tf.complex( tf.cast( spec.shape[-1], tf.float32), 0.)
    if offset is not None:
      window = tf.math.pow(
        tf.math.sin(mypi*offset + mypi*(end-offset)*tf.range(TD, dtype=tf.float32)/tf.cast(TD, tf.float32))
        ,power)
    else:
      window = tf.ones(shape=(TD,), dtype=tf.float32)
      
    window = tf.pad( window, [[0, spec_time.shape[-1] - TD]], "CONSTANT", constant_values=0.)
    window = tf.complex( tf.expand_dims( window, axis=0 ), 0.)
    #
    spec = tf.signal.fft( spec_time * window )
    return tf.math.real(spec), TD
  
  class MyModel(tf.keras.Model):
    
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # Scaling of FIDNet / refine within FIDNet
      self.MyScaling = tf.Variable( \
                                  initial_value = tf.constant(0.0, dtype=tf.float32),
                                  name='Scaling',
                                  trainable=False)
      
      #self.LossScaling = tf.Variable( \
      #                                initial_value=tf.constant(0.0, dtype=tf.float32),
      #                                name='LossScaling',
      #                                trainable=False)
      
    def train_step(self, data):
      (time, inp ), tar = data
      #
      # time    = (None,5,32768)
      # inp     = (None,  32768)
      # tar     = (None,5,32768)
      #
      with tf.GradientTape() as tape:
        #
        #
        y_pred = self((time,inp,self.MyScaling), training=True) 

        y_pred_spec = y_pred[...,0] # actual spectrum
        y_pred_conf = y_pred[...,1] # confidences (to be translated to sigma,esd below )
        #
        # First convolve spectra and uncertainties
        tar, _         = convolve2( tar, time,    transpose=True, offset=0.40)
        y_pred_spec, _ = convolve2( y_pred_spec, time,    transpose=True, offset=0.40)
        y_pred_conf, _ = convolve2( y_pred_conf, time,    transpose=True, offset=0.40)
        #
        sigma = tf.math.scalar_mul(0.998, tf.math.sigmoid(y_pred_conf)) + tf.constant( 0.001, dtype=y_pred_spec.dtype)
        sigma = tf.math.reciprocal_no_nan(sigma)
        sigma = tf.math.subtract( sigma, tf.constant(1., dtype=sigma.dtype))
        sigma = tf.math.scalar_mul(0.5, sigma )
        #
        y_pred_spec = tf.math.scalar_mul(0.1, y_pred_spec)
        tar         = tf.math.scalar_mul(0.1, tar)
        #
        y_pred = tf.stack( [y_pred_spec, sigma], axis=-1)
        #
        loss = self.compiled_loss( tf.math.scalar_mul(1.0, tar ) , \
                                   tf.math.scalar_mul(1.0, y_pred  ) )
        scaled_loss = self.optimizer.get_scaled_loss( loss )

      scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
      gradients = self.optimizer.get_unscaled_gradients( scaled_gradients )
      #gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      self.compiled_metrics.update_state( tf.math.scalar_mul(1.0, tar  ) , \
                                          tf.math.scalar_mul(1.0, y_pred ) )

      return {m.name: m.result() for m in self.metrics}

    def get_config(self):
      return {}
  #
  number_of_signals = 50
  MyScaling = 1.0
  JHH = (8.,3)
  NP=16384
  MinNP=12268
  #
  # read status:
  if os.path.isfile(StatusFile):
    for l in open(StatusFile):
      its=l.split()
      if its[0]=='number_of_signals':
        number_of_signals = int(its[1])
      if its[0]=='MyScaling':
        MyScaling = float(its[1])
      if its[0]=='JHH':
        temp0 = float(its[1])
        temp1 = float(its[2])
        JHH=(temp0,temp1)
      if its[0]=='NP':
        NP = int(its[1])
      if its[0]=='MinNP':
        MinNP=int(its[1])

    print(f'==========================================')         
    print(f' INFO: Reading parameters ')
    print(f'==========================================') 
    print(f' INFO: Signals   = {number_of_signals} ')
    print(f' INFO: MyScaling = {MyScaling} ')
    print(f' INFO: JHH       = {JHH[0] :.3f} ± {JHH[1] :.3f} ')
    print(f' INFO: NP        = {NP} ')
    print(f' INFO: MinNP     = {MinNP} ')
    print(f'==========================================') 
    #

  inputs = (tf.keras.Input(shape=(5,2*NP)),     \
            tf.keras.Input(shape=(5,2*NP)),     \
            tf.keras.Input(shape=() )              )
  outputs = fidnet_1d_model([inputs[0], inputs[1], inputs[2] ])
  model = MyModel(inputs=inputs, outputs=outputs)
  model.compile(optimizer=optimizer, loss=loss_function, metrics=[loss_value, loss_sigma_3, loss_sigma_4 ] )
  #model.MyScaling = tf.constant(MyScaling, dtype=tf.float32)
  model.MyScaling.assign(MyScaling)
  
  if Eval:
    # read from status file
    #number_of_signals=25
    pass

  for round in range(ROUNDS):
    print(f' #\n # Start round {round + 1}\n #')
    sys.stdout.flush()
    #
    # First make training planes
    #
    if Eval:
      nsigs = tf.random.uniform(minval=number_of_signals//2, maxval=number_of_signals, shape=(1000,), dtype=tf.dtypes.int32)  # Array with number of signals
    else:
      nsigs = tf.random.uniform(minval=3, maxval=number_of_signals, shape=(8000,), dtype=tf.dtypes.int32)  # Array with number of signals
      
    ds = tf.data.Dataset.from_tensor_slices( nsigs )
    if Eval:
      ds = ds.map( lambda x: MakeTraining(NSignals=x, NP=NP, MinNP=MinNP, JHH=JHH, Condense1H=False, Roofing=0.04, MaxCouplings=3, Noise=0.1, Eval=True), num_parallel_calls=2) # tf.data.experimental.AUTOTUNE )
      ds = ds.batch(1, drop_remainder=True)
    else:
      #
      ds = ds.map( lambda x: MakeTraining(NSignals=x, NP=NP, MinNP=MinNP, JHH=JHH, Roofing=0.04, MaxCouplings=3 ), num_parallel_calls= tf.data.experimental.AUTOTUNE )      
      #ds = ds.batch(len(tf.config.list_physical_devices('GPU')), drop_remainder=True)
      ds = ds.batch(BATCHSIZE, drop_remainder=True)

    if Eval:
      pass
    else:
      ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    class SaveLoss(tf.keras.callbacks.Callback):
        def __init__(self):
            super(tf.keras.callbacks.Callback,self).__init__()

        def on_epoch_end(self,epoch, logs):
            logfile=open(checkpoint_path+'/loss.out','a')
            logfile.write('%5d %10.5e %10.5e \n' %(epoch,logs['loss'], logs['loss_value'] ))
            logfile.flush()
            logfile.close()

    if Eval:
      counter=0
      losses=[]
      signals=[]

      # let's check for negative signals
      """
      for elem in ds:
        time = elem[0][0]
        inp = elem[0][1]
        SW   = elem[5][0,0].numpy()
        jhhs = elem[2]
        css  = elem[3]
        spins= elem[4]
        jhhs_mask = elem[6]
        
        inp_ft = tf.concat( [ inp[0,:,:], tf.zeros(shape=(inp.shape[1],inp.shape[-1]),dtype=inp.dtype)], axis=-1)
        inp_ft = tf.signal.fftshift( tf.signal.fft( tf.complex( inp_ft[:,0::2], inp_ft[:,1::2])),axes=-1)
        inp_ft = tf.math.real(inp_ft)
        inp_ft, _ = convolve2( inp_ft,        time,    transpose=False, offset=0.50)
        freqs = np.linspace(-SW/2., SW/2., inp_ft.shape[-1])
        plt.plot(freqs, inp_ft[0,:].numpy()/10., 'g-')
        ylims = plt.gca().get_ylim()
        #
        # Annotate spectrum
        print(' ======================= ')
        for i in range(nsigs[counter]):
          freqs=[]
          freqs.append( [css[0,0,i,0,0,0].numpy()] )
          theseJ = np.sort(jhhs[0,0,i,:,0,0].numpy())[::-1]
          theseJ = jhhs[0,0,i,:,0,0].numpy()
          
          print(' peak at ', css[0,0,i,0,0,0].numpy())
          print( jhhs[0,0,i,:,0,0].numpy() )
          print( spins[0,0,:].numpy() )
          print('')
          
          for j in range(nsigs[counter]):
            for _ in range( int(spins[0,0,j].numpy()+1.1) ):
              thisJ = theseJ[j] #jhhs[0,0,i,j,0,0].numpy()
              if np.abs(thisJ)>1.e-3:
                tt = []
                for f in freqs[-1]:
                  tt.append( f + thisJ/2. )
                  tt.append( f - thisJ/2. )
                freqs.append( tt )

          #plot frequencies

          for k,f in enumerate(freqs):
            if k==0:
              plt.plot( [ f[0], f[0] ], [ylims[0]*(1.+0.05*k),ylims[0]*(1.0+0.05*(k+0.5))], 'k-')
            else:
              #
              # Make angled lines
              for l in range(len(freqs[k-1])):
                plt.plot( [freqs[k-1][l], freqs[k][l*2  ]], [ylims[0]*(1.+0.05*(k-0.5)), ylims[0]*(1.+0.05*(k))], 'k-')
                plt.plot( [freqs[k-1][l], freqs[k][l*2+1]], [ylims[0]*(1.+0.05*(k-0.5)), ylims[0]*(1.+0.05*(k))], 'k-')
              for l in range(len(freqs[k])):
                plt.plot( [ freqs[k][l], freqs[k][l] ], [ylims[0]*(1.+0.05*(k)), ylims[0]*(1.+0.05*(k+0.5))], 'k-')

        plt.show()
      exit(10)
      """  
      for elem in ds:
        time = elem[0][0]
        inp  = elem[0][1]
        tar  = elem[1]
        jhhs = elem[2]
        css  = elem[3]
        spins= elem[4]
        SW   = elem[5][0,0].numpy()
        jhhs_mask = elem[6]
        #
        ypred_spec=[]
        ypred_esd =[]
        
        model.MyScaling.assign(1.0)
	# inference as below, or as model.predict([time,inp,model.MyScaling])
        ypred = model([time,inp,model.MyScaling], training=False)
        ypred_spec.append( ypred[...,0] )
        ypred_esd.append(  ypred[...,1] )
        
        inp_ft = tf.concat( [ inp[0,:,:], tf.zeros(shape=(inp.shape[1],inp.shape[-1]),dtype=inp.dtype)], axis=-1)
        inp_ft = tf.signal.fftshift( tf.signal.fft( tf.complex( inp_ft[:,0::2], inp_ft[:,1::2])),axes=-1)
        #inp_ft = tf.math.real(tf.expand_dims(inp_ft, axis=0))
        inp_ft = tf.math.real(inp_ft)
        
        tar_conv, _       = convolve2( tar,           time,    transpose=False, offset=0.40)
        ypred_esd[0], _   = convolve2( ypred_esd[0],  time,    transpose=False, offset=0.40)
        ypred_spec[0], _  = convolve2( ypred_spec[0], time,    transpose=False, offset=0.40)
        inp_ft, _         = convolve2( inp_ft,        time,    transpose=False, offset=0.40)

        #print( tf.math.reduce_max( tar_conv, axis=-1))
        loss_ = tf.math.reduce_mean(tf.square( 0.1*tar_conv - 0.1*ypred_spec[0]))

        print(f' Loss = {loss_.numpy() :.4f} counter = {counter} signals = {int(nsigs[counter]) } ')
        losses.append( loss_.numpy() )
        signals.append( int(nsigs[counter]))

        counter+=1

        freqs = np.linspace(-SW/2., SW/2., tar_conv.shape[-1])
          
        plt.plot(freqs, tar_conv[0,:].numpy()/10., 'b-', alpha=0.5)
        plt.plot(freqs, ypred_spec[0][0,:].numpy()/10., 'r-', alpha=0.5)
        plt.plot(freqs, (ypred_spec[0][0,:].numpy() - tar_conv[0,:].numpy())/10. - 20, 'k-')
        
        for i in range(1):
          plt.plot(freqs, inp_ft[i,:].numpy()/10.-40*(i+1), 'g-')
        #
        # assess uncertainties
        sigma = tf.math.scalar_mul(0.998, tf.math.sigmoid(ypred_esd[0])) + tf.constant( 0.001, dtype=ypred_spec[0].dtype)
        sigma = tf.math.reciprocal_no_nan(sigma)
        sigma = tf.math.subtract( sigma, tf.constant(1., dtype=sigma.dtype))
        sigma = tf.math.scalar_mul(0.5, sigma )

        ax1 = plt.gca()
        ax1.fill_between( freqs, sigma[0,:].numpy() -10., -sigma[0,:].numpy()-10., color='blue')
        #plt.plot( freqs,  simga[0,:] -10., 'g-')
        #plt.plot( freqs, -simga[0,:] -10., 'g-')

        plt.figure(2)

        diff_list = 0.1*(tar_conv[0] - ypred_spec[0] )
        sigma_list= sigma
        #
        # Reshape 
        sigma_list = tf.reshape( sigma_list, (-1,1))
        diff_list  = tf.reshape( diff_list,  (-1,1))
        #
        sigma_parts = tf.reshape( tfp.stats.quantiles( sigma_list, 100 ), (1,-1))
        #
        # Indices with sigma parts
        idxs = tf.where( ( sigma_parts[:,:-1] < sigma_list ) & \
                         ( sigma_parts[:,1:]  > sigma_list ), tf.constant(1.,dtype=tf.float32) , tf.constant(0.,dtype=tf.float32) ) #(5*400*512,sigma_parts)

        mean_diff_list  = tf.math.reduce_mean(diff_list)
        diff_list_tiles = tf.tile( diff_list, (1,sigma_parts.shape[-1]-1)) # (:, sigma_parts)
        counts = tf.math.reduce_sum( idxs, axis=0, keepdims=True ) #(1,sigma_parts)
        #
        calc_sigma = tf.math.reduce_sum( tf.square( (diff_list_tiles - mean_diff_list) * idxs ), axis=0, keepdims=True)/counts
        calc_sigma = tf.math.sqrt( calc_sigma )
        #
        pred_sigma = tf.math.reduce_sum( sigma_list * idxs, axis=0, keepdims=True )/counts  # (-1,1) * (-1,100) / (1,100)
        #
        sigma_loss_4 = tf.reduce_mean(tf.square(pred_sigma - calc_sigma))

        plt.plot( calc_sigma[0,:], pred_sigma[0,:], 'r.')
        line = np.linspace(0., np.max( tf.concat( [calc_sigma, pred_sigma], axis=0).numpy() ), 10,endpoint=True)
        plt.plot( line, line, 'k--')
        plt.xlabel(' calc sigma')
        plt.ylabel(' pred sigma')
        
        plt.show()
        #sys.exit(10)

            
    history = model.fit(ds,            \
                        epochs=EPOCHS, \
                        verbose=1,     \
                        callbacks=[SaveLoss()]  )

    if np.isnan(history.history['loss'][0]):
      print(f' Caught a NaN is the loss')
      print(f' INFO: Restore checkpoint at {ckpt_save_path}')      
      ckpt.restore(ckpt_manager.latest_checkpoint)
      continue
    
    ckpt_save_path = ckpt_manager.save()
    print(f' INFO: Saving checkpoint at {ckpt_save_path}')
    loss = history.history['loss'][0]
    #lr   = model.optimizer._optimizer._decayed_lr(tf.float32).numpy()
    lr   = model.optimizer._optimizer.lr.numpy()
    #
    # Slowly approach desired model and parameters
    ParamsUpdated=False
    if history.history['loss_value'][0]< (0.66 - (number_of_signals-10)*0.003):
      ParamsUpdated=True
      #
      # Scalar couplings
      temp=[JHH[0],JHH[1]]
      temp[0] = temp[0] - 0.095
      if temp[0] < 8.: temp[0]=8.0
      temp[1] = temp[1] + 0.011
      if temp[1] > 5.0: temp[1] = 5.0
      JHH=(temp[0],temp[1])
      #
      # Number of points in spectra
      if NP<16384:
        NP = NP + 32
        MinNP = MinNP + 24
        NP = np.min([NP,16*1024])
        MinNP = np.min([MinNP,16*(512+256)])
        #
        MyScaling=model.MyScaling
        inputs = (tf.keras.Input(shape=(5,2*NP)),     \
                  tf.keras.Input(shape=(5,2*NP)),     \
                  tf.keras.Input(shape=()  )          )
        outputs = fidnet_1d_model([inputs[0], inputs[1], inputs[2] ])
        model = MyModel(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=[loss_value ] )
        #
        # Read in the last checkpoint file      
        ckpt.restore(ckpt_manager.latest_checkpoint)
        model.MyScaling.assign(MyScaling) # do not think this is caught by the checkpointer
      #
      #Scaling in model
      if model.MyScaling.numpy()<1.0:
        model.MyScaling.assign_add(0.005)
      if model.MyScaling.numpy()>1.0:
        model.MyScaling.assign(1.0)

      #if (NP % 128 ) == 0:
      #  number_of_signals = number_of_signals + 1

    if history.history['loss_value'][0]<0.010:
       number_of_signals = number_of_signals + 1
       number_of_signals = np.min([number_of_signals,100])
      
    print(f' ==========================================') 
    if ParamsUpdated:
      print(f' INFO: Updated Parameters ')
    else:
      print(f' INFO: Current Parameters ')
    print(f' ==========================================') 
    print(f' INFO: learning_rate = {lr :10.5e} ')
    print(f' INFO: Signals       = {number_of_signals} ')
    print(f' INFO: MyScaling     = {model.MyScaling.numpy() :.4f} ')
    print(f' INFO: JHH           = {JHH[0] :.2f} ± {JHH[1] :.2f} ')
    print(f' INFO: NP,MinNP      = {NP}, {MinNP}')
    print(f' INFO: Next update at loss_value < {(0.66 - (number_of_signals-10)*0.003) :.4f}')
    print(f' ==========================================')
    
    loss_array.append(loss)
    lr_array.append(lr)
    # Write status file
    with open(StatusFile,'w') as ofs:
      ofs.write(f'number_of_signals {number_of_signals} \n')
      ofs.write(f'MyScaling {model.MyScaling.numpy() } \n')
      ofs.write(f'JHH {JHH[0]} {JHH[1]} \n')
      ofs.write(f'NP {NP} \n')
      ofs.write(f'MinNP {MinNP} \n')
      
    np.save( open(checkpoint_path + '/loss.npy','wb'), np.array(loss_array))
    np.save( open(checkpoint_path + '/lr.npy','wb')  , np.array(lr_array))
