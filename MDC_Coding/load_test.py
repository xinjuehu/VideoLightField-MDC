import numpy
from matplotlib.pyplot import imread
import re
import math
import torch
import dgl
import os
import matplotlib.image as mpimg
import sys
import math

from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import dgl.function as fn
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def SamplePattern(files_num,Pattern):
        SampleMatrix = torch.zeros(files_num,files_num,dtype=torch.double)
        for i in range(int((files_num - 1) / Pattern)):
            SampleMatrix[Pattern * (i + 1)-1, Pattern * (i + 1)-1] = 1
        SampleMatrix[files_num-1,files_num-1]=1
        return SampleMatrix

def AddEdges(g,sample_matrix,filesnum,Pattern):
        size = int(filesnum**0.5)
        set = numpy.linspace(-Pattern+1,Pattern-1,num=2*Pattern-1)
        likely = numpy.diag(sample_matrix).reshape(size, size)
        for i in range(size):
            for j in range(size):
                if likely[i, j] == 1:
                    g.add_edges(size * i + j, size * i + j)
                    continue
                for block_i in set:
                    block_i = int(block_i)
                    for block_j in set:
                        block_j = int(block_j)
                        new_i = i + block_i
                        new_j = j + block_j
                        if block_i==block_j==0 or new_i<0 or new_i>(size-1) or new_j<0 or new_j>(size-1):
                            continue
                        if likely[new_i,new_j]==1:
                            g.add_edges(size*new_i+new_j,size*i+j)
        return g

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

        """Update the node feature hv with ReLU(Whv+b)."""
    def forward(self, node):
        h = self.linear(node.data['h'].t())
        h = self.activation(h).t()
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature.t()
        msg = fn.copy_src(src='h', out='m')
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Reconstruction(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Reconstruction, self).__init__()
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu)])
            # GCN(hidden_dim, hidden_dim, F.relu)])
        self.reconstruction = nn.Linear(hidden_dim, out_dim)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        device = torch.device("cuda:0")
        h = g.ndata['h'].to(device)
        for conv in self.layers:
            h = h.t()
            h = conv(g, h)
        g.ndata['h'] = h
        # hg = dgl.mean_nodes(g, 'h')
        return self.reconstruction(g.ndata['h'].t()).t()

def reduce(nodes):
        """Take an average over all neighbor node features hu and use it to
        overwrite the original node feature."""
        accum = torch.mean(nodes.mailbox['m'], 1)
        return {'h': accum}


def per_test(frame_num,pattern_select):

    #frame_num = sys.argv[1]
    #pattern_select = sys.argv[2]
    #file_path = '/home/sjm/LFMPC_hxj/dataset/cats/split_matrix/frame_1/'
    file_path = '/home/sjm/LFMPC_hxj/dataset/chess2/resplit/frame_'+str(frame_num)+'/'
    #file1_path = '/home/sjm/LFMPC_hxj/cats_out/P2/frame_1_out/'
    file1_path = '/home/sjm/LFMPC_hxj/chess2_out/P'+str(pattern_select)+'/frame_'+str(frame_num)+'_out/'
    model_path = '/home/sjm/LFMPC_hxj/chess2_out/P'+str(pattern_select)+'/LF-GNN.pt'
    ref_file = '/home/sjm/LFMPC_hxj/dataset/chess2/split_matrix/frame_'+str(frame_num)+'/'
    #file1_path = sys.argv[2]
    dist_file = file1_path

    print(file_path)
    print(file1_path)


    for files in os.listdir(file_path):
        img = mpimg.imread(file_path+files)
        [x,y,z] = numpy.shape(img)
        print('inputfile_channel:')
        print(z)
        break

    numfrm = len(os.listdir(file_path))
    width = x
    height = y
    G = dgl.DGLGraph()
    G.add_nodes(numfrm)
    G.ndata['h'] = torch.zeros((numfrm, width*height),dtype=torch.double)
    input2 = torch.zeros((numfrm, width*height),dtype=torch.double)
    input3 = torch.zeros((numfrm, width*height),dtype=torch.double)

    #Read data
    i=0
    for files in os.listdir(file_path):
        img = mpimg.imread(file_path+files)
        img = img.astype('float64')
        img = torch.from_numpy(img)
        G.nodes[[i]].data['h'] = img[:,:,0].reshape(1,width*height)
        input2[i,:] = img[:, :, 1].reshape(1, width * height)
        input3[i,:] = img[:, :, 2].reshape(1, width * height)
        i = i + 1


    if not os.path.exists(file1_path):
        os.mkdir(file1_path)

    Pattern = pattern_select
    

    sample_matrix = SamplePattern(numfrm,Pattern)
    

    G = AddEdges(G,sample_matrix,numfrm,Pattern)

    label = G.ndata['h']
    input = sample_matrix.mm(G.ndata['h'])
    input2 = sample_matrix.mm(input2)
    input3 = sample_matrix.mm(input3)

    train_size = input.shape[1]
    batch_size = 10000

    msg = fn.copy_src(src='h', out='m')

    

    

    device = torch.device("cuda:0")


    model = torch.load(model_path)

    G.ndata['h'] = input
    prediction = model(G)
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()

    G.ndata['h'] = input2
    prediction2 = model(G)
    prediction2 = prediction2.cpu()
    prediction2 = prediction2.detach().numpy()

    G.ndata['h'] = input3
    prediction3 = model(G)
    prediction3 = prediction3.cpu()
    prediction3 = prediction3.detach().numpy()

    prediction[prediction>255]=255
    prediction[prediction<0]=0
    prediction2[prediction2>255]=255
    prediction2[prediction2<0]=0
    prediction3[prediction3>255]=255
    prediction3[prediction3<0]=0

    i = 0
    for files in os.listdir(file_path):
        img = numpy.zeros((width,height,3))
        img_r = prediction[i,:]
        img_g = prediction2[i,:]
        img_b = prediction3[i,:]
        img[:, :, 0] = img_r.reshape(width,height)
        img[:, :, 1] = img_g.reshape(width, height)
        img[:, :, 2] = img_b.reshape(width, height)
        img = img.astype('uint8')
        i = i + 1
        mpimg.imsave(file1_path+files,img)


    PSNR = numpy.zeros(5*5)
    SSIM = numpy.zeros(5*5)

    i = 0
    for files in os.listdir(ref_file):
        ref = mpimg.imread(ref_file+files)
#        ref = imread(ref_file+files).astype(numpy.float32)
#        ref = ref[:,:,range(3)]
        dist = mpimg.imread(dist_file+files)
#        dist = imread(dist_file+files).astype(numpy.float32)
        dist = dist[:,:,range(3)]
        PSNR[i] = compare_psnr(ref, dist)
        SSIM[i] = compare_ssim(ref, dist, multichannel = True)
        i = i + 1

    psnr_mat = PSNR.reshape(5,5)
    ssim_mat = SSIM.reshape(5,5)

    frame_scale = 5
    frame_num = frame_scale**2
    all_one = numpy.ones((frame_scale,frame_scale))
    likely = numpy.diag(sample_matrix).reshape(frame_scale, frame_scale)
    rec_psnr = (all_one-likely)*psnr_mat
    rec_ssim = (all_one-likely)*ssim_mat
    average_psnr = (numpy.sum(PSNR))/frame_num
    average_ssim = (numpy.sum(SSIM))/frame_num
    generated_psnr = (numpy.sum(rec_psnr))/(numpy.sum(all_one-likely))
    psnr_std = rec_psnr[rec_psnr>0].std()
    generated_ssim = (numpy.sum(rec_ssim))/(numpy.sum(all_one-likely))
    ssim_std = rec_ssim[rec_ssim>0].std()

    print(psnr_mat)
    print('generated_psnr:',generated_psnr)
    print('average_psnr:',average_psnr)
    print('psnr_std:',psnr_std)
    print(ssim_mat)
    print('generated_ssim:',generated_ssim)
    print('average_ssim:',average_ssim)
    print('ssim_std:',ssim_std)

b=16
for a in range(0,20):
    per_test(a,b)
