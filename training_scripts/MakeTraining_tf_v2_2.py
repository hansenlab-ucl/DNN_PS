import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as     tf
#import numpy as np
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#
# Input size (batch_size, fid_length, features)
# Features include [time, real, imag]
#
import numpy as np


def MakeTraining(NSignals=20,                 \
                 SFRQ=(400.,800.),           \
                 SWp=(9.,15.),               \
                 NP=4*1024,                  \
                 MinNP=4*(512+256),          \
                 R2=(3.,2.),                 \
                 JHC=(180.,30.),             \
                 JHH=(10.,4.),               \
                 Phase=(0.,1.),              \
                 Noise=(0.05,0.025),         \
                 Ints=(1,0.5),               \
                 BatchSize=1,                \
                 Eval=False,                 \
                 Roofing=None,               \
                 MaxCouplings=4,             \
                 SpinProb=(0.6,0.2,0.2),     \
                 Condense1H=False,           \
                 R2_inhom = (1.5, 1)
               ):

  BatchSize=1
  #
  #
  assert isinstance( SFRQ, (tuple, float, int))
  assert isinstance( SWp, (tuple, float, int))
  assert isinstance( Noise, (bool, float, tuple))
  #
  if isinstance( SFRQ, tuple ): 
    SFRQ = tf.random.uniform( shape=(1,), minval=tf.reduce_min(SFRQ), maxval=tf.reduce_max(SFRQ))
  elif isinstance( SFRQ, int ):
    SFRQ = tf.constant( SFRQ, dtype=tf.dtypes.float32)
  elif isinstance( SFRQ, float):
    SFRQ = tf.constant( SFRQ, dtype=tf.dtypes.float32)
  
  if isinstance( SWp, tuple ): 
    SWp = tf.random.uniform( shape=(1,), minval=tf.reduce_min(SWp), maxval=tf.reduce_max(SWp))
  elif isinstance( SWp, int ):
    SWp = tf.constant( SWp, dtype=tf.dtypes.float32)
  elif isinstance( SWp, float):
    SWp = tf.constant( SWp, dtype=tf.dtypes.float32)
  #
  if isinstance( Noise, tuple ):
    Noise_stds = tf.random.normal( shape=(1,), mean=Noise[0], stddev=Noise[1], dtype=tf.dtypes.float32)
    Noise_stds = tf.abs(Noise_stds)
  elif isinstance( Noise, float ):
    Noise_stds = tf.constant( Noise, dtype=tf.dtypes.float32)
  elif Noise==False:
    Noise_stds=tf.constant(0.0, dtype=tf.dtypes.float32)
  else:
    sys.stderr.write(' Noise has to be <float>, <tuple>(max, min), <bool>(False) \n')
    sys.exit(10)    
  #
  # We first generate the signal in the direct dimension
  SW = SWp * SFRQ
  SW = tf.random.normal( shape=(1,), mean=10000., stddev=1000., dtype=SW.dtype)
  time = tf.range(NP, dtype=tf.dtypes.float32)/SW
  #
  pi180 = tf.math.asin(1.)/tf.constant(90.,dtype=tf.float32) # PI/180.
  twopi = tf.math.asin(1.)*tf.constant(4., dtype=tf.float32) # 2*pi
  mypi  = tf.math.asin(1.)*tf.constant(2., dtype=tf.float32) # pi
  #
  if Roofing is not None:
    Roofing = tf.constant( Roofing, dtype=tf.float32)
  #
  if Condense1H and False:
    # shall we condense (20% of the times)
    condense = tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0)
    loc = tf.cond( condense > 0.80, lambda: tf.random.uniform(shape=(1,), minval=-0.9*SW, maxval=0.9*SW ), lambda: 0.0 )
    css = tf.cond( condense > 0.80, \
                   lambda: tf.random.normal(shape=(BatchSize,NSignals,1,1,1), mean=loc, stddev=0.10*SW, dtype=tf.dtypes.float32), \
                   lambda: tf.random.normal(shape=(BatchSize,NSignals,1,1,1), mean=0.0, stddev=0.25*SW, dtype=tf.dtypes.float32) )
    css = tf.math.tanh(css*2./SW)*SW*0.5
  else:
    # print(NSignals.shape)
    # sys.exit(10)
    num_condense = NSignals//2

    if num_condense >= 10:
    # if False:
      num_non_condense = NSignals - num_condense
      loc = tf.random.uniform(shape=(1,), minval=-0.7*SW, maxval=0.7*SW)
      scale_css = tf.random.uniform(shape=(1,), minval=0.01, maxval=0.05)
      css_condense = tf.random.normal(shape=(BatchSize,NSignals,1,1,1), mean=loc, stddev=scale_css*SW, dtype=tf.dtypes.float32)
      css_condense = css_condense[:,:num_condense, :,:,:]
      css_non_condense = tf.random.normal(shape=(BatchSize,NSignals,1,1,1), mean=0., stddev=0.25*SW, dtype=tf.dtypes.float32)
      css_non_condense = css_non_condense[:,:num_non_condense, :,:,:]
     # css_non_condense = tf.math.tanh(css_non_condense*2./SW)*SW*0.5 
      css  = tf.concat([css_condense, css_non_condense], axis=1)
      css = tf.math.tanh(css*2./SW)*SW*0.5 # Make sure we stay within SW

    else:

      css = tf.random.normal(shape=(BatchSize,NSignals,1,1,1), mean=0., stddev=0.25*SW, dtype=tf.dtypes.float32)
      css = tf.math.tanh(css*2./SW)*SW*0.5 # Make sure we stay within SW

  r2s = tf.abs(tf.random.normal(mean=R2[0], stddev=R2[1], shape=(BatchSize,NSignals, 1,1,1), dtype=tf.dtypes.float32))
  r2_inhom = tf.abs(tf.random.normal(mean=R2_inhom[0], stddev=R2_inhom[1], shape=(BatchSize,1, 1,1,1), dtype=tf.dtypes.float32)) 
  # print(r2_inhom.shape)
  # sys.exit(10)  
  # r2_inhom = tf.abs(tf.random.normal(mean=R2_inhom[0], stddev=R2_inhom[1], shape=(BatchSize,NSignals, 1,1,1), dtype=tf.dtypes.float32))  
  jchs= tf.random.normal(mean=JHC[0],  stddev=JHC[1],  shape=(BatchSize,NSignals,1,1,1), dtype=tf.dtypes.float32)
  #
  #only 90% with satelites
  jchs = tf.where( tf.random.uniform(shape=(BatchSize,NSignals,1,1,1), dtype=jchs.dtype, minval=0., maxval=1.0)>0.90, 0., jchs )

  phis= tf.random.normal(mean=Phase[0],stddev=Phase[1],shape=(BatchSize,NSignals,1,1,1), dtype=tf.dtypes.float32)*pi180
  ints= tf.random.normal(mean=Ints[0], stddev=Ints[1], shape=(BatchSize,NSignals,1,1,1), dtype=tf.dtypes.float32)
  ints= tf.math.abs(ints)
  #
  # How many spins per site:
  p = tf.math.log( tf.reshape(tf.constant(SpinProb,dtype=tf.float32),(1,-1)) )
  spins = tf.random.categorical( p, NSignals ) # n-1 so the power for the cosine
  spins = tf.cast( spins, tf.float32)
  ints = (spins[:,:,tf.newaxis,tf.newaxis,tf.newaxis]+1.)*ints
  
  jhhs= tf.random.normal(mean=JHH[0],  stddev=JHH[1],  shape=(BatchSize,NSignals,NSignals,1), dtype=tf.dtypes.float32)
  #
  # Make symmetric
  jhhs = 0.5*(jhhs + tf.transpose(jhhs, perm=(0,2,1,3)))
  #
  #
  # Make jhh mask. These are classical daisy chains (with cross-overs)
  # if MaxCouplings=3, if nsigs=odd, then the middle spin will have coups=2 and the rest 3
  #                    for nsigns=even, then all spins will have coups=3
  ones = tf.ones(shape=(NSignals,NSignals), dtype=tf.float32)
  if MaxCouplings==2:
    mask_j2 = tf.linalg.band_part( ones, 1, 1 ) - tf.linalg.diag( tf.ones(shape=(NSignals,))) + \
              tf.ones(shape=(NSignals,NSignals), dtype=tf.float32) - tf.linalg.band_part( ones, NSignals-2, NSignals-2 )
    jhhs_mask = mask_j2
  elif MaxCouplings==3:
    mask_j3 = tf.linalg.band_part( ones, 1, 1 ) - tf.linalg.diag( tf.ones(shape=(NSignals,))) + \
              tf.roll( tf.linalg.diag( tf.ones(shape=NSignals,)), (NSignals+1)//2, axis=-1) + \
              tf.ones(shape=(NSignals,NSignals), dtype=tf.float32) - tf.linalg.band_part( ones, NSignals-2, NSignals-2 )
    jhhs_mask = mask_j3
  elif MaxCouplings==4:
    mask_j4 = tf.linalg.band_part( ones, 2, 2 ) - tf.linalg.diag( tf.ones(shape=(NSignals,))) + \
              tf.ones(shape=(NSignals,NSignals), dtype=tf.float32) - tf.linalg.band_part( ones, NSignals-3, NSignals-3 )
    jhhs_mask = mask_j4
  else:
    print(f' MaxCouplings={MaxCouplings} is not implemented', file=sys.stderr)
    print(' Use MaxCouplings of 2 or 4', file=sys.stderr)
    print(' .MakeTraining() ', file=sys.stderr)
    sys.stderr.flush()
    sys.exit(10)
  #
  # chop some of the couplings out
  jhhs_mask = tf.linalg.band_part( jhhs_mask, 0, -1 ) # upper triangular part
  mask_sel  = tf.random.uniform(minval=0., maxval=1., dtype=tf.float32, shape=(NSignals,NSignals) )
  jhhs_mask = tf.where( mask_sel < 0.40, 0., jhhs_mask)
  jhhs_mask = jhhs_mask + tf.transpose(jhhs_mask, perm=(1,0))
  #
  jhhs_mask = jhhs_mask[tf.newaxis,:,:,tf.newaxis]
  #
  jhhs = jhhs * jhhs_mask
  #
  #
  # Make target - no couplings
  r2s_tar = r2s
  fids_tar = tf.complex(ints,0.) * tf.math.exp(
    tf.complex(0.,phis) +
    tf.complex( -r2s_tar*time, css*twopi*time))      
  fids_tar = tf.math.reduce_sum( fids_tar, axis=(1,2,3))
 
  #
  #
  # Lets check for strong(est) couplings. We pick the three spins with lowest highest J/DeltaCS
  if False:
    diff = tf.reshape(css, (-1,1)) - tf.reshape(css, (1,-1))  #DeltaCS
    diff = tf.abs(diff/(0.01 + jhhs[0,:,:,0]))                #DeltaCS/J
    print(' original diff \n', diff.numpy(), '\n')
    print(' original jhhs \n', jhhs[0,:,:,0].numpy(), '\n')
    #
    #
    # We always take the three combinations with smallest diff
    diffp= 1./(diff+1e5*tf.eye(diff.shape[-1]))
    #
    print(' J/DeltaCS \n', diffp.numpy(), '\n')
    print('max(J/DeltaCS)\n', tf.math.reduce_max(diffp, axis=-1))
    
    #diffpp = tf.reshape( diffp, (-1,))
    #vals, idxs = tf.math.top_k( diffpp, 6)
    #
    #dd = tf.expand_dims(vals, axis=0) - tf.expand_dims( diffp, axis=-1)
    #dd = tf.math.reduce_min( tf.abs(dd), axis=-1)
    #
    #sc_jhhs = tf.where( tf.abs(dd)<1.0e-6, jhhs[0,:,:,0], 0. )
    #sc_jhhs = sc_jhhs[tf.newaxis,:,:,tf.newaxis]
    #
    # Now eliminate these couplings in the original jhhs
    #jhhs = tf.where( tf.abs(dd)<1.0e-6, 0., jhhs[0,:,:,0])
    #jhhs = jhhs[tf.newaxis,:,:,tf.newaxis]
    #
    #print(' Stripped JHHS \n', jhhs[0,:,:,0].numpy(), '\n' )
    #print(' Strong coupling \n', sc_jhhs[0,:,:,0].numpy(), '\n' )
    #
    # We ned to setup the Liouvillian
    
    
    #
    sys.exit(10)
  #
  # We have 3 input fids with times [ 0ms, 10ms, 30ms, 50, 90ms ]
  add_evol = tf.constant([0.000, 0.010, 0.030, 0.050, 0.090 ], dtype=tf.dtypes.float32)

  time_gen = time[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]  # (Batch, n_signals, 1, time_pts, add_evol)
  time_gen = time_gen + tf.reshape( add_evol, (1, 1, 1, 1, -1))

  jhhs = jhhs[:,:,:,:, tf.newaxis]  #( batch, n_signals, n_signals, 1, 1
  #
  cos_term = tf.complex(tf.math.cos( mypi * jhhs * time_gen ), 0. )
  sin_term = tf.complex(tf.math.sin( mypi * jhhs * time_gen ), 0. )

  if Roofing is not None:
    roofing = tf.random.normal(shape=(NSignals,NSignals), mean=0.0, stddev=Roofing ) #tf.ones( shape=(NSignals, NSignals) )
    roofing = tf.linalg.band_part( roofing, 0, -1 ) - tf.linalg.band_part( roofing, 0, 0 )
    roofing = tf.complex(0., roofing)
    roofing = roofing + tf.math.conj( tf.transpose(roofing, perm=(1,0)))
    roofing = roofing[tf.newaxis,:,:,tf.newaxis,tf.newaxis]
  else:
    roofing = tf.zeros( shape=cos_term.shape, dtype=tf.complex64)
  #
  # This holds the cosine evolution for all signals with all times - and also the sine term for roofing effects
  cos_term = cos_term + roofing*sin_term
  #
  # add number of spins in active spin
  power_term = tf.complex(1.+spins[tf.newaxis,:,:,tf.newaxis,tf.newaxis],0.)
  cos_term = tf.pow( cos_term, power_term )
  #
  # quite sure that we should multiply by zero below for the sin_term (this is now included above)
  cos_term = tf.math.reduce_prod( \
                                  cos_term + 0.*roofing*sin_term \
                                  , axis=2, keepdims=True ) #(batch, n_signals, 1, time, add)
  #
  # this holds the 13C satelites
  cos_sat = 0.99 + 0.01*tf.math.cos( mypi * jchs * time[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis] )
  
  # generate fids
  fids_inp = tf.complex(ints,0.) *               \
             cos_term *        \
             tf.complex( cos_sat, 0.) *         \
             tf.math.exp( tf.complex(0., phis) + \
                          tf.complex(0., css*2*mypi * time[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis] ) - \
                          tf.complex(r2s*time_gen,0.))

  fids_inp = tf.math.reduce_sum( fids_inp, axis=(1,2))

  # print(fids_inp)

  #
  # add noise
  fid_noise = tf.complex( \
                             tf.random.normal(mean=0., stddev=Noise_stds, shape=fids_inp.shape, dtype=tf.float32), \
                             tf.random.normal(mean=0., stddev=Noise_stds, shape=fids_inp.shape, dtype=tf.float32) )
  fids_inp += fid_noise 

  pts_to_use_H = tf.math.scalar_mul(tf.constant(1, dtype=tf.dtypes.float32),
                                    tf.math.floor(tf.random.uniform(shape=(1,), minval=MinNP, maxval=NP, dtype=tf.dtypes.float32)))
  
  pts_to_use_H = tf.expand_dims(pts_to_use_H, axis=-1)
  pts_mask_h   = tf.where( tf.expand_dims( tf.range(NP, dtype=tf.dtypes.float32), axis=0 ) < pts_to_use_H, 1., 0.) #(1,512)

  fids_tar = fids_tar * tf.complex( pts_mask_h, 0.)
  fids_inp = fids_inp * tf.complex( pts_mask_h[:,:,tf.newaxis], 0.)
  time     = time_gen[:,0,0,:,:] * pts_mask_h[:,:,tf.newaxis]
  
  time = tf.tile( time, (BatchSize, 1, 1))
  time = tf.transpose( time, (0,2,1)) # (batch, coup, :)
  fids_inp = tf.transpose( fids_inp, (0,2,1)) # (batch, coup, :)

  time=      tf.reshape(tf.stack([time,time],             axis=-1),(BatchSize,time.shape[1] ,2*NP))  # (batch,coup,(time.r,time.i))
  fids_inp = tf.reshape(tf.stack([tf.math.real(fids_inp), tf.math.imag(fids_inp)], axis=-1), (BatchSize,fids_inp.shape[1], 2*NP))


  fids_tar = tf.reshape(tf.stack([tf.math.real(fids_tar), tf.math.imag(fids_tar)], axis=-1), (BatchSize, 2*NP))

  # #### add inhomgeneity


  time_mult = tf.reshape(time[0][0], shape=fids_tar.shape)
  mult = tf.math.exp(tf.math.multiply(-1*float(r2_inhom[0,0,0,0,0]), time_mult))
  fids_tar = tf.math.multiply(fids_tar, mult)
  mult = tf.reshape(tf.stack([mult,mult, mult, mult, mult]), shape = (1, 5, 2*NP))

  fids_inp = tf.math.multiply(fids_inp, mult) 

  # time_mult = tf.reshape(time[0][0], shape=fids_tar.shape)
  
  # fids_tar = tf.math.multiply(fids_tar, tf.math.exp(tf.math.multiply(0.5, time_mult)))

  # print(fids_tar)
  #
  # Normalise each batch
  norm_factor = tf.reduce_max( tf.abs(fids_inp), axis=(1,2) )
  #
  #
  # normalise to 1k points
  TD = 0.5*(tf.reduce_max(tf.where( time > 0., tf.range(time.shape[-1], dtype=tf.float32), 0.))+1)
  #
  fids_inp = ((1024.)/TD)*tf.cast(NSignals,tf.float32)*fids_inp/norm_factor
  fids_tar = ((1024.)/TD)*tf.cast(NSignals,tf.float32)*fids_tar/norm_factor
  #
  #
  if not Eval:
    time    = tf.cast(time, tf.float32)
    fids_inp= tf.cast(fids_inp,tf.float32)

  #
  # We need to FT the target
  # Zero fill
  ft_tar = tf.pad( fids_tar, [ [0,0], [0,fids_tar.shape[-1]]], "CONSTANT", constant_values=0.)
  multiplier = tf.concat( [ tf.constant([0.5,], dtype=ft_tar.dtype), tf.ones( (ft_tar.shape[-1]//2-1,), dtype=ft_tar.dtype)], axis=0)
  multiplier = tf.complex( tf.reshape( multiplier, (1,-1)), tf.constant(0.,dtype=multiplier.dtype))
  #
  ft_tar = tf.signal.fftshift( tf.signal.fft( \
                                              multiplier*tf.complex(ft_tar[...,0::2], ft_tar[...,1::2]) \
  ) , axes=-1)
  ft_tar = tf.math.real(ft_tar) 

  time = tf.squeeze( time, axis=0)
  fids_inp = tf.squeeze( fids_inp, axis=0)
  fids_tar = tf.squeeze( fids_tar, axis=0)
  ft_tar   = tf.squeeze( ft_tar,   axis=0)
  
  if Eval:
    return ((time, fids_inp), ft_tar, jhhs, css, spins, SW, jhhs_mask )
  else:
    return ((time, fids_inp), ft_tar )    

