import numpy as np
import math

def cal_sending_seq(presentation_num, packet_num):
    # IF_list = np.array([13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2])
    IF_list = np.array(range(presentation_num))


    Sending_list = np.array(range(presentation_num))
    Is_boud = np.zeros(presentation_num)
    for i in range(presentation_num):
        packet_id_start = np.floor((packet_num * i)/presentation_num).astype(int)
        packet_id_end = np.floor((packet_num * (i+1))/presentation_num).astype(int)
        if packet_id_start != packet_id_end:
            Is_boud[i] = 1
    
    arrange_position = 101
    last_position = 100
    for j in range(presentation_num):
        if last_position == 0:
            if Is_boud[last_position+1] != 2:
                arrange_position = last_position + 1
                Sending_list[IF_list[j]] = arrange_position
                Is_boud[arrange_position] = 2
                last_position = arrange_position
                continue
        if (last_position == presentation_num - 1):
            if Is_boud[last_position-1] != 2:
                arrange_position = last_position - 1
                Sending_list[IF_list[j]] = arrange_position
                Is_boud[arrange_position] = 2
                last_position = arrange_position
                continue
        if last_position >0 and last_position < presentation_num -1:
            if Is_boud[last_position+1] != 2:
                arrange_position = last_position + 1
                Sending_list[IF_list[j]] = arrange_position
                Is_boud[arrange_position] = 2
                last_position = arrange_position
                continue
            if Is_boud[last_position-1] != 2:
                arrange_position = last_position - 1
                Sending_list[IF_list[j]] = arrange_position
                Is_boud[arrange_position] = 2
                last_position = arrange_position
                continue 
        if np.array(np.where(Is_boud == 0)).size != 0:
            waiting_list = np.array(np.where(Is_boud == 0))[0]
            arrange_position = waiting_list[0]
            # for k in waiting_list:
            #     if abs(k - last_position) > 1:
            #         arrange_position = k
            #         break
            #     else:
            #         arrange_position = waiting_list[0]
        else:
            waiting_list = np.array(np.where(Is_boud == 1))[0]
            arrange_position = waiting_list[0]
            # for k in waiting_list:
            #     if abs(k - last_position) > 1:
            #         arrange_position = k
            #         break
            #     else:
            #         arrange_position = waiting_list[0]
        Sending_list[IF_list[j]] = arrange_position
        Is_boud[arrange_position] = 2
        last_position = arrange_position
    
    return Sending_list


# a= 500/1024
# b = 0.15 * 64 /32
# c = b*16/a
# result_test = cal_sending_seq(16,c)
# print(result_test)