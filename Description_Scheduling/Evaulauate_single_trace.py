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

os.remove('random.csv')
os.remove('scheduled.csv')

OUT_PUT_PSNR = np.array([(0, 1, 2, 4, 8, 16), (0, 32.5674, 36.1864, 39.722, 41.0232, 44.4218)])
Presentatiom_BANDWIDTH = (241.575/1000) * 64 /16
#Presentatiom_BANDWIDTH = 0.05 * 64 /16
Packet_size = 500/1024
Packet_loss_ration = 0.1

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



highest_level_1 = np.zeros(Prensentation_num.shape)
output_psnr_1 = np.zeros(Prensentation_num.shape)
output_psnr_1.dtype = 'float'
highest_level_2 = np.zeros(Prensentation_num.shape)
output_psnr_2 = np.zeros(Prensentation_num.shape)
output_psnr_2.dtype = 'float'
for i in range(Prensentation_num.shape[0]):
	Sending_Seq_1 = np.arange(Prensentation_num[i])
	np.random.shuffle(Sending_Seq_1)
	Received_Presentation = Packetloss_sim(Presentatiom_BANDWIDTH, Prensentation_num[i], Packet_loss_ration, Sending_Seq_1, Packet_size)
	highest_level_1[i] = VP_aggregation(Received_Presentation)
	
	Packet_num = np.ceil(Prensentation_num[i] * Presentatiom_BANDWIDTH / Packet_size).astype(int)
	Sending_Seq_2 = cal_sending_seq(Prensentation_num[i], Packet_num)
	Received_Presentation = Packetloss_sim(Presentatiom_BANDWIDTH, Prensentation_num[i], Packet_loss_ration, Sending_Seq_2, Packet_size)
	highest_level_2[i] = VP_aggregation(Received_Presentation)
	for level_k in range(6):
		if highest_level_1[i] == OUT_PUT_PSNR[0,level_k]:
			output_psnr_1[i] = OUT_PUT_PSNR[1,level_k] + np.random.rand()
		if highest_level_2[i] == OUT_PUT_PSNR[0,level_k]:
			output_psnr_2[i] = OUT_PUT_PSNR[1,level_k] + np.random.rand()

b1=np.savetxt('random.csv',output_psnr_1,fmt='%1.3f',delimiter=None)
b2=np.savetxt('scheduled.csv',output_psnr_2,fmt='%1.3f',delimiter=None)

# plt.figure(1)
# plt.plot(np.array(time_all[1:]) / MILLISECONDS_IN_SECONDS, 
# 		 highest_level_1)

# plt.figure(2)
# plt.plot(np.array(time_all[1:]) / MILLISECONDS_IN_SECONDS, 
# 		 highest_level_2)

# plt.figure(3)
# plt.plot(np.array(time_all[1:]) / MILLISECONDS_IN_SECONDS, 
# 		 Prensentation_num)

# plt.show()

print(highest_level_1)
Rate_1 = np.zeros(6)
Rate_1[0] = (np.array(np.where(highest_level_1==16)).shape[1])/highest_level_1.shape[0]
Rate_1[1] = (np.array(np.where(highest_level_1==8)).shape[1])/highest_level_1.shape[0]
Rate_1[2] = (np.array(np.where(highest_level_1==4)).shape[1])/highest_level_1.shape[0]
Rate_1[3] = (np.array(np.where(highest_level_1==2)).shape[1])/highest_level_1.shape[0]
Rate_1[4] = (np.array(np.where(highest_level_1==1)).shape[1])/highest_level_1.shape[0]
Rate_1[5] = (np.array(np.where(highest_level_1==0)).shape[1])/highest_level_1.shape[0]
print(Rate_1)

Rate_2 = np.zeros(6)
Rate_2[0] = (np.array(np.where(highest_level_2==16)).shape[1])/highest_level_2.shape[0]
Rate_2[1] = (np.array(np.where(highest_level_2==8)).shape[1])/highest_level_2.shape[0]
Rate_2[2] = (np.array(np.where(highest_level_2==4)).shape[1])/highest_level_2.shape[0]
Rate_2[3] = (np.array(np.where(highest_level_2==2)).shape[1])/highest_level_2.shape[0]
Rate_2[4] = (np.array(np.where(highest_level_2==1)).shape[1])/highest_level_2.shape[0]
Rate_2[5] = (np.array(np.where(highest_level_2==0)).shape[1])/highest_level_2.shape[0]
print(Rate_2)



