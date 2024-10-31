import os,sys
  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
import tensorflow as     tf
import nmrglue as ng
import numpy as np 
import logging 
from PureShift.FIDNet import *
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--spin_echo", type=Path)
parser.add_argument("--psyche", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument('--phase', type = int, choices =[0, 1], default=0)
parser.add_argument('--p0', type = float, default=0.0)
parser.add_argument('--p1', type = float, default=0.0)
parser.add_argument('--pp0', type = float, default=0.0)
parser.add_argument('--pp1', type = float, default=0.0)
parser.add_argument('--clear_phase', type = int, choices =[0, 1], default=0)
parser.add_argument('--scaling', type = int, default=20)
parser.add_argument('--ver_scale', type = float, default=1.0)

args = parser.parse_args()

logging.getLogger('tensorflow').setLevel(logging.ERROR)
checkpoint_path = "./PureShift/weights/"

gpu = False

if gpu:
  strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
  os.environ['CUDA_VISIBLE_DEVICES']="0"
else:
  strategy = tf.distribute.OneDeviceStrategy(device='/cpu')

with strategy.scope():

  fidnet_1d_model = OneDFIDNet(
    fidnet_filters=48,
    blocks=3,
    fidnet_kernel=(1,16),
    refine_kernel=(1,16),
    dilations=[1,2,4,6,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256],
    rate=0.10 )

  ckpt = tf.train.Checkpoint(transformer=fidnet_1d_model)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('\n INFO:  Model weights restored \n')

  def convolve2(spec, time, transpose=False, offset=0.40, end=0.98, power=2.0):
    spec = hilbert(spec)
    mypi = tf.math.asin(1.)*2.

    spec_time = tf.math.conj(tf.signal.fft( spec ))/tf.complex( tf.cast( spec.shape[-1], tf.float32), 0.)

    myrange = tf.range(time.shape[-1], dtype=tf.int32)
    myrange = tf.reshape( myrange, (1,-1))
    myrange = tf.tile( myrange, (time.shape[0],1))

    TD = tf.cast(tf.reduce_max(tf.where( time[:,0,...] > 0., myrange, 0), axis=-1)+1, tf.dtypes.int32)//2

    if offset is not None:
      myrange = tf.reshape( tf.range(time.shape[-1]//2, dtype=tf.float32), (1,-1))
      window = tf.math.pow(
          tf.math.sin(3.1415*offset + 3.1415*(end-offset)*tf.cast(myrange,tf.float32)/tf.expand_dims(tf.cast(TD,tf.float32),axis=-1))
      ,power)
    else:
      window = tf.ones(shape=(time.shape[0],time.shape[-1]//2), dtype=tf.float32)

    myrange = tf.reshape( tf.range(time.shape[-1]//2, dtype=tf.int32), (1,-1))
    window = tf.where( myrange < tf.reshape(TD, (-1,1)), window, 0.)
    window = tf.pad( window, [ [0,0], [0,window.shape[-1]]], "CONSTANT", constant_values=0.)
    window = tf.complex( window, 0.)

    spec = tf.signal.fft( spec_time * window )
    return tf.math.real(spec), TD
  
  class MyModel(tf.keras.Model):
    
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.MyScaling = tf.Variable( \
                                  initial_value = tf.constant(0.0, dtype=tf.float32),
                                  name='Scaling')
 
  NP=16384
  inputs = (tf.keras.Input(shape=(5,2*NP)),     \
            tf.keras.Input(shape=(5,2*NP)),     \
            tf.keras.Input(shape=() )              )
  outputs = fidnet_1d_model([inputs[0], inputs[1], inputs[2] ])

  model = MyModel(inputs=inputs, outputs=outputs)
  model.MyScaling.assign(1.0)

def time_proc(uc, inp):

    increment = [0, 0.010, 0.030, 0.050, 0.09]
    time_li = []
    for i in increment:
        time_li.append(i+uc.sec_scale()[:16384])  
    
    time = tf.reshape(np.array(time_li), shape=(1, 5, 16384))
    time = tf.reshape(tf.stack([time,time],axis=-1),(1,time.shape[1] ,2*16384))  # (batch,coup,(time.r,time.i))
    inp = tf.reshape(tf.stack([tf.math.real(inp), tf.math.imag(inp)], axis=-1), (1, 5, 16384*2))

    return time, inp

def normalize(inp, time):
    
    norm_factor = tf.reduce_max( tf.abs(inp), axis=(1,2))
    TD = 0.5*(tf.reduce_max(tf.where( time > 0., tf.range(time.shape[-1], dtype=tf.float32), 0.))+1)
    inp = ((256.)/TD)*tf.cast(50,tf.float32)*tf.cast(inp, tf.float32)/tf.cast(norm_factor, tf.float32)

    return TD, tf.cast(norm_factor, tf.float32), tf.cast(inp, tf.float32)


def prediction(model, dic, inp, time, norm_factor, lw = 2, 
               pred_scale=0.25, vertical = 20, psyche_1d=None, psyche_scale=None):
    
    ypred_spec=[]
    ypred_esd =[]
    
    model.MyScaling.assign(1.0)
    ypred = model([time,inp,model.MyScaling], training=False)
    ypred_spec.append( ypred[...,0] )
    ypred_esd.append(  ypred[...,1] )
    
    inp_ft = tf.concat( [ inp[0,:,:], tf.zeros(shape=(inp.shape[1],inp.shape[-1]),dtype=inp.dtype)], axis=-1)
    inp_ft = tf.signal.fftshift( tf.signal.fft( tf.complex( inp_ft[:,0::2], inp_ft[:,1::2])),axes=-1)
    inp_ft = tf.math.real(inp_ft)
    
    if psyche_1d is not None:
      tar_conv, _       = convolve2( psyche_1d,     time,    transpose=False, offset=0.40)
    else:
      pass
    ypred_esd[0], _   = convolve2( ypred_esd[0],  time,    transpose=False, offset=0.40)
    ypred_spec[0], _  = convolve2( ypred_spec[0], time,    transpose=False, offset=0.40)
    inp_ft, _         = convolve2( inp_ft,        time,    transpose=False, offset=0.40)
    

    lw = np.exp(-np.pi*time[0][0,:]*lw)
    
    sign = np.fft.ifft(ypred_spec[0][0]) * lw                     
    ypred_spec_em = np.fft.fft(sign).real
    
    if psyche_1d is not None:
    
      target = np.fft.ifft(tar_conv[0]) * lw
      target = np.fft.fft(target).real
      
    else:
      pass
      
    SW = dic['acqus']['SW']
    car = dic['acqus']['O1']/dic['acqus']['SFO1']
    freqs = np.linspace(car-SW/2., car+SW/2., 16384)
      
    org_1d = np.fft.ifft(inp_ft[0])
    org_1d = np.fft.fft(org_1d.real*lw + 1j * \
                               org_1d.imag*lw).real[::2]

    sigma = tf.math.scalar_mul(0.998, tf.math.sigmoid(ypred_esd[0])) + tf.constant( 0.001, dtype=ypred_spec[0].dtype)
    sigma = tf.math.reciprocal_no_nan(sigma)
    sigma = tf.math.subtract( sigma, tf.constant(1., dtype=sigma.dtype))
    sigma = tf.math.scalar_mul(0.5, sigma )
    
    if psyche_1d is not None:
      return freqs, sigma[0].numpy()[::2], ypred_spec_em[::2], target[::2], org_1d, norm_factor
    else:
      return freqs, sigma[0].numpy()[::2], ypred_spec_em[::2], org_1d, norm_factor
      
      
def spinecho_data_import(file_name, p0, p1):
    
    dic, data = ng.bruker.read(file_name)
    data[:,0] = data[:,0]*0.5
    data = ng.bruker.remove_digital_filter(dic, data)
    data = ng.proc_base.zf_size(data, 16384) 
    data = ng.proc_base.fft(data)
    data = ng.proc_base.ps(data, p0=p0, p1=p1)
    data = ng.proc_base.ifft(data)
    data = ng.proc_bl.cbf(data)
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileiobase.uc_from_udic(udic)
    mod = data[:,:16384]
    mod = tf.convert_to_tensor(mod)
    inp = tf.reshape(mod, shape=(1, 5, 16384))

    return dic, inp, uc
    
    
def process_psyche(file_bruker, norm_factor, TD, p0, p1):
    
    dic1, data = ng.bruker.read(file_bruker)
    data = ng.bruker.remove_digital_filter(dic1, data)
    data = ng.proc_base.zf_size(data, 16384*2) 
    data = ng.proc_base.fft(data)
    data = ng.proc_base.ps(data, p0=p0, p1=p1)
    psyche_1d = ng.proc_base.di(data)
    psyche_1d = ((256.)/TD)*tf.cast(50,tf.float32)*psyche_1d/norm_factor
    psyche_1d = tf.reshape(psyche_1d, shape=(1, 16384*2))
    
    return psyche_1d
    
    
def save_output(predicted, folder_name):

  if args.psyche is not None:
    files = ['freqs', 'uncertainty', 'predicted', 'psyche', 'conventional', 'norm_factor']
  else:
    files = ['freqs', 'uncertainty', 'predicted', 'conventional', 'norm_factor']
  
  for i in range(len(predicted)):
    if files[i] != 'norm_factor':
      np.savetxt(folder_name+'/'+files[i]+".csv", predicted[i], delimiter=",")
      
def plot_spectra(predicted, scaling, linewidth = 0.6, ver_scale = 0.25):

    if args.psyche is not None:
      freqs, sigma, pred, psyche, conv = [i for i in predicted[:-1]]
      
    else:
      freqs, sigma, pred, conv = [i for i in predicted[:-1]]
    
    fig, ax = plt.subplots(figsize=(12, 12)) # for full
    
    pred_up = sigma
    pred_low = -sigma
    
    if args.psyche is not None:
      ax.plot(freqs, 4*ver_scale*conv, lw=linewidth, color='blue', label = f'Conventional 1H : x{4*4*ver_scale}')
      ax.plot(freqs, scaling+(psyche*64*ver_scale), lw=linewidth, color = 'green', label = f'PSYCHE 1H : x{64*4*ver_scale}')
      ax.fill_between(freqs, (2*scaling)+(4*ver_scale*pred_low), (2*scaling)+(4*ver_scale*pred_up), color = 'darkred', alpha=0.6, lw=linewidth, label = f'Uncertainty : x{4*4*ver_scale}')
      ax.plot(freqs, (3*scaling)+(ver_scale*pred), lw=linewidth, color='red', label = f'DNN pureshift 1H : x{4*ver_scale}')
      ax.fill_between(freqs,(3*scaling)+(ver_scale*(pred+pred_low)), (3*scaling)+(ver_scale*(pred+pred_up)), color = 'black', alpha=0.6, lw=linewidth)
    else:
      ax.plot(freqs, 4*ver_scale*conv, lw=linewidth, color='blue', label = f'Conventional 1H : x{4*4*ver_scale}')
      ax.fill_between(freqs, (2*scaling)+(4*ver_scale*pred_low), (2*scaling)+(4*ver_scale*pred_up), color = 'darkred', alpha=0.6, lw=linewidth, label = f'Uncertainty : x{4*4*ver_scale}')
      ax.plot(freqs, (3*scaling)+(ver_scale*pred), lw=linewidth, color='red', label = f'DNN pureshift 1H : x{4*ver_scale}')
      ax.fill_between(freqs,(3*scaling)+(ver_scale*(pred+pred_low)), (3*scaling)+(ver_scale*(pred+pred_up)), color = 'black', alpha=0.6, lw=linewidth)
        
    ax.set_xlim(freqs.max(), freqs.min())
    plt.xticks(size=16)
    plt.yticks([])
    ax.minorticks_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel('$^{1}$H (ppm)', size=20)
    plt.savefig('DNN_pureshift.pdf')
    plt.legend()
    plt.show()      
