import numpy as np

def Packetloss_sim(Presentation_size, Presentation_num, loss_ratio, Presentation_Seq, Packet_size):
    Packet_num = (np.ceil(Presentation_size * Presentation_num / Packet_size)).astype(int)
    loss_packet_num = (np.ceil(Packet_num * loss_ratio)).astype(int)
    loss_packet_list = np.random.choice(Packet_num, loss_packet_num, replace=False)
    Presentation_received_array = np.zeros(Presentation_num) 
    for i in range(Presentation_num):
        packet_id_start = np.floor((i*Presentation_size)/Packet_size).astype(int)
        packet_id_end = np.ceil(((i+1)*Presentation_size)/Packet_size).astype(int)
        taget_packet = np.array(range(packet_id_start,packet_id_end))
        if np.sum(np.in1d(taget_packet, loss_packet_list)) == 0:
            Presentation_received_array[i] = 1
    Received_Presentation = Presentation_Seq[np.where(Presentation_received_array>0)]
    return Received_Presentation
