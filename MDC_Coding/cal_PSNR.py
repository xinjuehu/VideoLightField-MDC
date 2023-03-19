import numpy
from matplotlib.pyplot import imread
import sys
import re
import os
import math

import videoquality.psnr as psnr

ref_file = '/home/sjm/LFMPC_hxj/dataset/cats/split_matrix/frame_0/'
#file1_path = sys.argv[2]
dist_file = '/home/sjm/LFMPC_hxj/dataset/cats/resplit/frame_0/'

PSNR = numpy.zeros(8*8)

i = 0
for files in os.listdir(ref_file):
    ref = imread(ref_file+files).astype(numpy.float32)
    dist = imread(dist_file+files).astype(numpy.float32)
    dist = dist[:,:,range(3)]
    PSNR[i] = psnr.psnr(ref, dist)
    i = i + 1

psnr_mat = PSNR.reshape(8,8)

Pattern = 2
frame_scale = 8
frame_num = frame_scale**2
SampleMatrix = numpy.ones((frame_num,frame_num))
for i in range(math.floor(frame_num / Pattern)):
    SampleMatrix[Pattern * (i + 1) -1, Pattern * (i + 1)-1] = 0
SampleMatrix[frame_num-1,frame_num-1]=0
likely = numpy.diag(SampleMatrix).reshape(frame_scale, frame_scale)
rec = likely*psnr_mat

gen_view_num = numpy.sum(likely)
average_psnr = (numpy.sum(rec) + 100*(64-gen_view_num))/64

print(likely)
print(average_psnr)
