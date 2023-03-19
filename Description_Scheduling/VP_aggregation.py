import numpy as np

def VP_aggregation(reveived_sequence):
    Seq = np.array(reveived_sequence)
    Seq = Seq.astype(int)
    sample_spatio_list = np.array([1,1/2,1/4,1/8,1/16])
    break_trigger = 0
    higest_level_ratio = 0
    for k in sample_spatio_list:
        presentation_list = generate_presentation(k, 16)
        for m in range(presentation_list.shape[0]):
            active_presentation = presentation_list[m,:]
            if np.sum(np.in1d(active_presentation, Seq)) == active_presentation.shape[0]:
                break_trigger = 1
                #print('The best aggregation combination is:')
                #print(active_presentation)
                break
        if break_trigger == 1:
            higest_level_ratio = k*16
            break
    return higest_level_ratio


def generate_presentation(sample_spatio, VP_num):
    presentation_num = 1/sample_spatio
    presentation_num = presentation_num.astype(int)
    presentation_VP = VP_num * sample_spatio
    presentation_VP = presentation_VP.astype(int)
    target_presentation = np.zeros([presentation_num, presentation_VP])
    for i in range(presentation_num):
        i_start = i*presentation_VP
        i_start = i_start.astype(int)
        i_end = (i+1)*presentation_VP
        i_end = i_end.astype(int)
        target_presentation[i,:] = np.array(range(i_start, i_end))
    target_presentation = target_presentation.astype(int)
    return target_presentation
