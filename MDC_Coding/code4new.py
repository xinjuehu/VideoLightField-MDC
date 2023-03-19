import numpy
import matplotlib.image as mpimg
import sys
import re
import shutil
from shutil import copyfile
import os
import math

raw_file_dir = './dataset/chess1/raw/'
ref_file_dir = './dataset/chess1/split_matrix/'
spl_file_dir = './dataset/chess1/encode/'
dec_file_dir = './dataset/chess1/decode/'
resplit_file_dir = './dataset/chess1/resplit/'

if os.path.exists(spl_file_dir):
    shutil.rmtree(spl_file_dir)
os.mkdir(spl_file_dir)

if os.path.exists(ref_file_dir):
    shutil.rmtree(ref_file_dir)
os.mkdir(ref_file_dir)

if os.path.exists(dec_file_dir):
    shutil.rmtree(dec_file_dir)
os.mkdir(dec_file_dir)

if os.path.exists(resplit_file_dir):
    shutil.rmtree(resplit_file_dir)
os.mkdir(resplit_file_dir)

frame_start = 1
 
#img = mpimg.imread(raw_file_dir+'frame_'+str(frame_start)+'/sa_'+str(1)+'_'+str(1)+'-pt-30-40'+'.png')
img = mpimg.imread(raw_file_dir+'frame_'+str(frame_start)+'/sa_'+str(1)+'_'+str(1)+'.png')
[x,y,z] = numpy.shape(img)


numfrm = 100
width = y
subwidth = int(y/4)
height = x
subheight = int(x/4)
channel = z





for row in range(5):
    for col in range(5):
        vp_dir = spl_file_dir + str(row*8+col)
        os.mkdir(vp_dir)
        Frame_id = 0
        sub_dir = vp_dir +'/'
        for frames in range(numfrm):
            img = mpimg.imread(raw_file_dir+'frame_'+str(Frame_id+frame_start)+'/sa_'+str(row+1)+'_'+str(col+1)+'.png')
            #img = mpimg.imread(raw_file_dir+'frame_'+str(Frame_id+frame_start)+'/sa_'+str(row+1)+'_'+str(col+1)+'-pt-30-40'+'.png')
            img_nam = "%03d" % (Frame_id)
            file_name = 'image'+img_nam+'.jpg'
            subaperture = numpy.zeros((subheight,subwidth,channel))
            subaperture = 255*img[0:-1:2,0:-1:2,:]
            subaperture = subaperture.astype('uint8')
            mpimg.imsave(sub_dir+file_name,subaperture)
            if not os.path.exists(ref_file_dir+'frame_'+str(Frame_id)+'/'):
                os.mkdir(ref_file_dir+'frame_'+str(Frame_id)+'/')
            new_spli_img = ref_file_dir+'frame_'+str(Frame_id)+'/'+str(row)+'_'+str(col)+'.jpg'
            mpimg.imsave(new_spli_img,subaperture)
            Frame_id = Frame_id + 1
        retval = os.getcwd()
        os.chdir(sub_dir)
        #os.system('ffmpeg -framerate 24 -i image%03d.jpg -c:v h264_nvenc -preset lossless -profile high output.mp4')
        os.system('ffmpeg -framerate 24 -i image%03d.jpg -c:v h264_nvenc -profile high output.mp4')
        os.chdir(retval)
        det_dir = dec_file_dir + str(row*8+col)
        os.mkdir(det_dir)
        sub2_dir = det_dir +'/'
        copyfile(sub_dir+'output.mp4', sub2_dir+'input.mp4')
        os.chdir(sub2_dir)
        os.system('ffmpeg -i input.mp4 -qscale:v 2 output_%03d.jpg')
        os.remove('input.mp4')
        os.chdir(retval)
        Frame_id = 0
        for files in os.listdir(sub2_dir):
            sub3_dir = resplit_file_dir+'frame_'+str(Frame_id)+'/'
            if not os.path.exists(sub3_dir):
                os.mkdir(sub3_dir)
            file_name = str(row)+'_'+str(col)+'.jpg'
            copyfile(sub2_dir+files, sub3_dir+file_name)
            Frame_id = Frame_id + 1

