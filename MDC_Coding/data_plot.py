import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set_style('darkgrid')
sns.set_context('paper')


def SamplePattern(frame_num,Pattern):
    SampleMatrix = np.ones((frame_num,frame_num))
    for i in range(math.floor(frame_num / Pattern)):
        SampleMatrix[Pattern * (i + 1) -1, Pattern * (i + 1)-1] = 0
    SampleMatrix[frame_num-1,frame_num-1]=0
    return SampleMatrix

def load_data(P,Pattern):
    P = np.array(P)
    sample_matrix = SamplePattern(25,Pattern)
    mask = np.diag(sample_matrix).reshape(5, 5)
    P = P*mask
    P = P.flatten()
    P = P[P>0]
    df = pd.DataFrame(P)
    return df


Pattern=2
P=[[35.31457556,37.70539889,35.47296153,37.80393502,34.39321993]
,[38.01137694,36.2475675,37.28863376,35.87700208,37.37165434]
,[34.63069822,37.7664085,35.72095295,37.39691348,35.10487256]
,[37.62330646,37.15521446,37.45375893,35.92074293,37.5058973,]
,[35.64026641,37.58184033,36.56640685,37.42867681,37.28288043]]
df2=load_data(P,Pattern)

Pattern=4
P=[[35.26439586,34.96656405,33.74041083,37.59761851,32.29639747]
,[29.70241965,35.4545693,37.55364522,31.85336854,34.3699236,]
,[34.5144402,37.73883144,33.69670174,30.60801889,34.51638927]
,[37.30275513,30.9296141,31.79238776,34.37201722,37.37964085]
,[35.54090383,34.33192871,31.62992722,37.58231655,37.59315769]]
df4=load_data(P,Pattern)

Pattern=8
P=[[27.02032569,34.60271018,31.68106069,25.84201157,29.42736081]
,[29.0561805,34.44396962,37.38270079,30.43069535,33.56531817]
,[31.06918835,26.36672912,27.59656856,30.36891329,31.04598237]
,[37.10894878,30.65424373,31.70139648,30.1305044,26.92855844]
,[26.78890558,28.86247351,31.46048413,37.45731154,37.35421172]]
df8=load_data(P,Pattern)

Pattern=16
P=[[24.2828627,33.33914525,29.67233782,24.37146462,28.44970805]
,[28.7991989,30.52020465,30.33067917,30.31554279,33.37840296]
,[25.91963108,24.42575069,26.00669269,29.89699138,26.32466895]
,[37.47540579,30.11386413,31.01227142,26.45883095,24.04958242]
,[24.40849803,26.32430616,30.91128065,29.24625552,37.26076848]]
df16=load_data(P,Pattern)




ax = sns.distplot(df2, bins = 10, hist = False, kde = True, norm_hist = False,
            rug = False, vertical = False,
            color = 'b', label = '1:2')
ax = sns.distplot(df4, bins = 10, hist = False, kde = True, norm_hist = False,
            rug = False, vertical = False,
            color = 'r', label = '1:4')
ax = sns.distplot(df8, bins = 10, hist = False, kde = True, norm_hist = False,
            rug = False, vertical = False,
            color = 'g', label = '1:8')
ax = sns.distplot(df16, bins = 10, hist = False, kde = True, norm_hist = False,
            rug = False, vertical = False,
            color = 'y', label = '1:16')
ax.set_xlabel('Viewpoint PSNR')
ax.set_ylabel('Percentage')
ax.set_title('Distribution of Decoded Viewpoint PSNR using Different Description')
plt.legend()
plt.show()