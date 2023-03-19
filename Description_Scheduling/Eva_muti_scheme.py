import numpy as np
import matplotlib.pyplot as plt
from Packetloss_Sim import Packetloss_sim
from VP_aggregation import VP_aggregation
from VP_aggregation import generate_presentation
from Sending_seq import cal_sending_seq
import os

mahimahi_packet = 1500.0  # bytes
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0
N = 100

OUT_PUT_PSNR = np.array([(0, 1, 2, 4, 8, 16), (0, 32.5674, 36.1864, 39.722, 41.0232, 44.4218)])
Presentatiom_BANDWIDTH = (241.575/1000) * 64 /16
#Presentatiom_BANDWIDTH = 0.05 * 64 /16
Packet_size = 500/1024
Packet_loss_ration = 0.3

MDCNN_P_Band_1 = 4.6279
MDCNN_P_Band_2 = 7.2631
MDCNN_P_Band_3 = 13.4201
MDCNN_P_Band_4 = 20.5947
MDCNN_P_Band_5 = 30.3543
MDCNN_P_Band_6 = 70.0984

MDCNN_PSNR = np.array([0,23.4242,25.5630,28.1444,29.9514,31.2664,37.3218])

SHEVC_P_Band_1 = 5.583
SHEVC_P_Band_2 = 8.2091
SHEVC_P_Band_3 = 12.02
SHEVC_P_Band_4 = 36.9508

SHEVC_PSNR_1 = 31.1205
SHEVC_PSNR_2 = 36.649
SHEVC_PSNR_3 = 38.7712
SHEVC_PSNR_4 = 44.6613

os.remove('our_result.csv')
os.remove('SHEVC_result.csv')
os.remove('time_stamp.csv')
os.remove('MDCNN_result.csv')
os.remove('throughput.csv')

#LINK_FILE = './mahimahi/test'
#LINK_FILE = './mahimahi/trace_4011749_https---www.youtube.com'
#LINK_FILE = './mahimahi/3(3G)/bus.ljansbakken-oslo/bus.ljansbakken-oslo.report.2010-09-28_1407CEST.log'
LINK_FILE = './mahimahi/4(4G)/\logs_all/report_bus_0001.log'

time_all = []
packet_sent_all = []
last_time_stamp = 0
packet_sent = 0
with open(LINK_FILE, 'rb') as f:
	for line in f:
		time_stamp = int(line.split()[0])
		if time_stamp == last_time_stamp:
			packet_sent += 1
			continue
		else:
			time_all.append(last_time_stamp)
			packet_sent_all.append(packet_sent)
			packet_sent = 1
			last_time_stamp = time_stamp

time_window = np.array(time_all[1:]) - np.array(time_all[:-1])
throuput_all = mahimahi_packet * \
			   BITS_IN_BYTE * \
			   np.array(packet_sent_all[1:]) / \
			   time_window * \
			   MILLISECONDS_IN_SECONDS / \
			   MBITS_IN_BITS

Real_throuput = np.convolve(throuput_all, np.ones(N,)/N, mode='same')
Prensentation_num = np.floor(Real_throuput/Presentatiom_BANDWIDTH).astype(int)
SHEVC_P_PSNR = np.zeros(Prensentation_num.shape[0])
MDCNN_P_PSNR = np.zeros(Prensentation_num.shape[0])

highest_level = np.zeros(Prensentation_num.shape)
our_psnr = np.zeros(Prensentation_num.shape)
our_psnr.dtype = 'float'

for i in range(Prensentation_num.shape[0]):
    if Real_throuput[i] > SHEVC_P_Band_4:
        P1 = np.random.rand()
        P2 = np.random.rand()
        P3 = np.random.rand()
        P4 = np.random.rand()
        SHEVC_P_PSNR[i] = SHEVC_PSNR_4
        if P4< Packet_loss_ration:
            SHEVC_P_PSNR[i] = SHEVC_PSNR_3
        if P3< Packet_loss_ration:
            SHEVC_P_PSNR[i] = SHEVC_PSNR_2
        if P2< Packet_loss_ration:
            SHEVC_P_PSNR[i] = SHEVC_PSNR_1
        if P1< Packet_loss_ration:
            SHEVC_P_PSNR[i] = 0
    if SHEVC_P_Band_3 < Real_throuput[i] < SHEVC_P_Band_4:
        P1 = np.random.rand()
        P2 = np.random.rand()
        P3 = np.random.rand()
        SHEVC_P_PSNR[i] = SHEVC_PSNR_3
        if P3< Packet_loss_ration:
            SHEVC_P_PSNR[i] = SHEVC_PSNR_2
        if P2< Packet_loss_ration:
            SHEVC_P_PSNR[i] = SHEVC_PSNR_1
        if P1< Packet_loss_ration:
            SHEVC_P_PSNR[i] = 0
    if SHEVC_P_Band_2 < Real_throuput[i] < SHEVC_P_Band_3:
        P1 = np.random.rand()
        P2 = np.random.rand()
        SHEVC_P_PSNR[i] = SHEVC_PSNR_2
        if P2< Packet_loss_ration:
            SHEVC_P_PSNR[i] = SHEVC_PSNR_1
        if P1< Packet_loss_ration:
            SHEVC_P_PSNR[i] = 0
    if SHEVC_P_Band_1 < Real_throuput[i] < SHEVC_P_Band_2:
        P1 = np.random.rand()
        SHEVC_P_PSNR[i] = SHEVC_PSNR_1
        if P1< Packet_loss_ration:
            SHEVC_P_PSNR[i] = 0
    if Real_throuput[i] < SHEVC_P_Band_1:
        SHEVC_P_PSNR[i] =0
    
    SHEVC_P_PSNR[i] = SHEVC_P_PSNR[i] + np.random.rand()

    if Real_throuput[i] > MDCNN_P_Band_6:
        P = np.random.rand(6)
        sucess_num = np.array(np.where(P>Packet_loss_ration)).shape[1]
        MDCNN_P_PSNR[i] = MDCNN_PSNR[sucess_num]
    if MDCNN_P_Band_5 < Real_throuput[i] < MDCNN_P_Band_6:
        P = np.random.rand(5)
        sucess_num = np.array(np.where(P>Packet_loss_ration)).shape[1]
        MDCNN_P_PSNR[i] = MDCNN_PSNR[sucess_num]
    if MDCNN_P_Band_4 < Real_throuput[i] < MDCNN_P_Band_5:
        P = np.random.rand(4)
        sucess_num = np.array(np.where(P>Packet_loss_ration)).shape[1]
        MDCNN_P_PSNR[i] = MDCNN_PSNR[sucess_num]
    if MDCNN_P_Band_3 < Real_throuput[i] < MDCNN_P_Band_4:
        P = np.random.rand(3)
        sucess_num = np.array(np.where(P>Packet_loss_ration)).shape[1]
        MDCNN_P_PSNR[i] = MDCNN_PSNR[sucess_num]
    if MDCNN_P_Band_2 < Real_throuput[i] < MDCNN_P_Band_3:
        P = np.random.rand(2)
        sucess_num = np.array(np.where(P>Packet_loss_ration)).shape[1]
        MDCNN_P_PSNR[i] = MDCNN_PSNR[sucess_num]
    if MDCNN_P_Band_1 < Real_throuput[i] < MDCNN_P_Band_2:
        P = np.random.rand(1)
        sucess_num = np.array(np.where(P>Packet_loss_ration)).shape[1]
        MDCNN_P_PSNR[i] = MDCNN_PSNR[sucess_num]
    if Real_throuput[i] < MDCNN_P_Band_1:
        MDCNN_P_PSNR[i] =0
    
    MDCNN_P_PSNR[i] = MDCNN_P_PSNR[i] + np.random.rand()
    
    Packet_num = np.ceil(Prensentation_num[i] * Presentatiom_BANDWIDTH / Packet_size).astype(int)
    Sending_Seq = cal_sending_seq(Prensentation_num[i], Packet_num)
    Received_Presentation = Packetloss_sim(Presentatiom_BANDWIDTH, Prensentation_num[i], Packet_loss_ration, Sending_Seq, Packet_size)
    highest_level[i] = VP_aggregation(Received_Presentation)
    for level_k in range(6):
        if highest_level[i] == OUT_PUT_PSNR[0,level_k]:
            our_psnr[i] = OUT_PUT_PSNR[1,level_k] + np.random.rand()
time_stamp_index = range(Prensentation_num.shape[0])
our_psnr_sub = our_psnr[0:-1:60]
SHEVC_P_PSNR_sub = SHEVC_P_PSNR[0:-1:60]
MDCNN_P_PSNR_sub = MDCNN_P_PSNR[0:-1:60]
time_stamp_index_sub = time_stamp_index[0:-1:60]
Real_throuput_sub = Real_throuput[0:-1:60]
b1=np.savetxt('our_result.csv',our_psnr_sub,fmt='%1.3f',delimiter=None)
b2=np.savetxt('SHEVC_result.csv',SHEVC_P_PSNR_sub,fmt='%1.3f',delimiter=None)
b4=np.savetxt('MDCNN_result.csv',MDCNN_P_PSNR_sub,fmt='%1.3f',delimiter=None)
b3=np.savetxt('time_stamp.csv',time_stamp_index_sub,fmt='%1.3f',delimiter=None)
b5=np.savetxt('throughput.csv',Real_throuput_sub,fmt='%1.3f',delimiter=None)