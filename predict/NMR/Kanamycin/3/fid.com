#!/bin/csh

bruk2pipe -verb -in ./ser \
  -bad 0.0 -ext -aswap -DMX -decim 2464 -dspfvs 20 -grpdly 67.9842376708984  \
  -xN             32768  -yN                 5  \
  -xT             16384  -yT                 5  \
  -xMODE            DQD  -yMODE           Real  \
  -xSW         8116.883  -ySW           50.104  \
  -xOBS         800.444  -yOBS         800.444  \
  -xCAR           4.702  -yCAR           4.772  \
  -xLAB             1Hx  -yLAB             1Hy  \
  -ndim               2  -aq2D       Magnitude  \
#| nmrPipe -fn MULT -c 9.76562e-01 \
  -out ./int.fid -ov

nmrPipe -verb -in int.fid                \                                  \
#   |   nmrPipe -fn SP -off 0.40 -end 0.98 -pow 2 -c 0.5 \
#   |   nmrPipe -fn ZF -auto                     \
 #  |   nmrPipe -fn FT -auto                    \
   |   nmrPipe -fn PS -p0 40.2 -p1 9.0       \
   |   nmrPipe -fn POLY -auto \
 #  |   nmrPipe -fn HT -auto \
 #  |   nmrPipe -fn FT -inv \
   |   nmrPipe -ov -verb -out test.fid 
endif

sleep 5
