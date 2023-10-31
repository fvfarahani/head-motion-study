# conda activate env_pytorch
#%% ===========================================================================
# PHASE I:
#     1. Identifying subjects who have all sessions
#     2. Extracting HCP data (ONE TIME)
#     3. Parcelating data (ONE TIME)
#     4. Save (ONE TIME)
# =============================================================================

import os, glob
import numpy as np

computer = 'local' # 'JHPCE' or 'local'

# Subjetcs in each set/session
if computer == 'JHPCE':
    os.chdir('/dcl01/smart/data/fvfarahani/HCP_Motion/RSmotion/')
else:
    os.chdir('/Volumes/Elements/HCP_Motion/RSmotion/')
files_REST1_LR = sorted(glob.glob('*REST1_LR*.txt'))
files_REST1_RL = sorted(glob.glob('*REST1_RL*.txt'))
files_REST2_LR = sorted(glob.glob('*REST2_LR*.txt'))
files_REST2_RL = sorted(glob.glob('*REST2_RL*.txt'))
ID_REST1_LR = set([int(files_REST1_LR[i][:6]) for i in range(len(files_REST1_LR))])
ID_REST1_RL = set([int(files_REST1_RL[i][:6]) for i in range(len(files_REST1_RL))])
ID_REST2_LR = set([int(files_REST2_LR[i][:6]) for i in range(len(files_REST2_LR))])
ID_REST2_RL = set([int(files_REST2_RL[i][:6]) for i in range(len(files_REST2_RL))])

if computer == 'JHPCE':
    os.chdir('/dcl01/smart/data/fvfarahani/HCP_Motion/WMmotion/')
else:
    os.chdir('/Volumes/Elements/HCP_Motion/WMmotion/')
files_WM_LR = sorted(glob.glob('*WM_LR*.txt'))
files_WM_RL = sorted(glob.glob('*WM_RL*.txt'))
ID_WM_LR = set([int(files_WM_LR[i][:6]) for i in range(len(files_WM_LR))])
ID_WM_RL = set([int(files_WM_RL[i][:6]) for i in range(len(files_WM_RL))])

if computer == 'JHPCE':
    os.chdir('/dcl01/smart/data/fvfarahani/HCP_Motion/GAMBLINGmotion/')
else:
    os.chdir('/Volumes/Elements/HCP_Motion/GAMBLINGmotion/')
files_GAMBLING_LR = sorted(glob.glob('*GAMBLING_LR*.txt'))
files_GAMBLING_RL = sorted(glob.glob('*GAMBLING_RL*.txt'))
ID_GAMBLING_LR = set([int(files_GAMBLING_LR[i][:6]) for i in range(len(files_GAMBLING_LR))])
ID_GAMBLING_RL = set([int(files_GAMBLING_RL[i][:6]) for i in range(len(files_GAMBLING_RL))])

if computer == 'JHPCE':
    os.chdir('/dcl01/smart/data/fvfarahani/HCP_Motion/MOTORmotion/')
else:
    os.chdir('/Volumes/Elements/HCP_Motion/MOTORmotion/')
files_MOTOR_LR = sorted(glob.glob('*MOTOR_LR*.txt'))
files_MOTOR_RL = sorted(glob.glob('*MOTOR_RL*.txt'))
ID_MOTOR_LR = set([int(files_MOTOR_LR[i][:6]) for i in range(len(files_MOTOR_LR))])
ID_MOTOR_RL = set([int(files_MOTOR_RL[i][:6]) for i in range(len(files_MOTOR_RL))])

if computer == 'JHPCE':
    os.chdir('/dcl01/smart/data/fvfarahani/HCP_Motion/LANGUAGEmotion/')
else:
    os.chdir('/Volumes/Elements/HCP_Motion/LANGUAGEmotion/')
files_LANGUAGE_LR = sorted(glob.glob('*LANGUAGE_LR*.txt'))
files_LANGUAGE_RL = sorted(glob.glob('*LANGUAGE_RL*.txt'))
ID_LANGUAGE_LR = set([int(files_LANGUAGE_LR[i][:6]) for i in range(len(files_LANGUAGE_LR))])
ID_LANGUAGE_RL = set([int(files_LANGUAGE_RL[i][:6]) for i in range(len(files_LANGUAGE_RL))])

if computer == 'JHPCE':
    os.chdir('/dcl01/smart/data/fvfarahani/HCP_Motion/SOCIALmotion/')
else:
    os.chdir('/Volumes/Elements/HCP_Motion/SOCIALmotion/')
files_SOCIAL_LR = sorted(glob.glob('*SOCIAL_LR*.txt'))
files_SOCIAL_RL = sorted(glob.glob('*SOCIAL_RL*.txt'))
ID_SOCIAL_LR = set([int(files_SOCIAL_LR[i][:6]) for i in range(len(files_SOCIAL_LR))])
ID_SOCIAL_RL = set([int(files_SOCIAL_RL[i][:6]) for i in range(len(files_SOCIAL_RL))])

if computer == 'JHPCE':
    os.chdir('/dcl01/smart/data/fvfarahani/HCP_Motion/RELATIONALmotion/')
else:
    os.chdir('/Volumes/Elements/HCP_Motion/RELATIONALmotion/')
files_RELATIONAL_LR = sorted(glob.glob('*RELATIONAL_LR*.txt'))
files_RELATIONAL_RL = sorted(glob.glob('*RELATIONAL_RL*.txt'))
ID_RELATIONAL_LR = set([int(files_RELATIONAL_LR[i][:6]) for i in range(len(files_RELATIONAL_LR))])
ID_RELATIONAL_RL = set([int(files_RELATIONAL_RL[i][:6]) for i in range(len(files_RELATIONAL_RL))])

if computer == 'JHPCE':
    os.chdir('/dcl01/smart/data/fvfarahani/HCP_Motion/EMOTIONmotion/')
else:
    os.chdir('/Volumes/Elements/HCP_Motion/EMOTIONmotion/')
files_EMOTION_LR = sorted(glob.glob('*EMOTION_LR*.txt'))
files_EMOTION_RL = sorted(glob.glob('*EMOTION_RL*.txt'))
ID_EMOTION_LR = set([int(files_EMOTION_LR[i][:6]) for i in range(len(files_EMOTION_LR))])
ID_EMOTION_RL = set([int(files_EMOTION_RL[i][:6]) for i in range(len(files_EMOTION_RL))])

# Subjects who have all sessions
ID = set.intersection(ID_REST1_LR, ID_REST1_RL, ID_REST2_LR, ID_REST2_RL,
                      ID_WM_LR, ID_WM_RL, ID_GAMBLING_LR, ID_GAMBLING_RL,
                      ID_MOTOR_LR, ID_MOTOR_RL, ID_LANGUAGE_LR, ID_LANGUAGE_RL,
                      ID_SOCIAL_LR, ID_SOCIAL_RL, ID_RELATIONAL_LR, ID_RELATIONAL_RL,
                      ID_EMOTION_LR, ID_EMOTION_RL)
# subtract from incomplete data (timeseries)
IN = set.union(set([119833, 140420, 284646]), # REST1_LR
                       set([119732, 150423, 159946, 169747, 183337, 751550]), # REST2_LR
                       set([119732, 169747, 171431, 183337, 317332, 751550, 786569]), # REST2_RL
                       set([179548, 650746]), # WM 
                       set([179548, 650746, 150423, 329440, 547046]), # GAMBLING --> RuntimeWarning: 144428
                       set([179548, 650746]), # MOTOR --> RuntimeWarning: 144428
                       set([179548, 650746, 713239]), # LANGUAGE
                       set([179548, 650746]), # SOCIAL
                       set([179548, 650746, 150423]), # RELATIONAL --> RuntimeWarning: invalid value encountered in true_divide: 223929
                       set([179548, 650746])) # EMOTION --> RuntimeWarning: 168139

ID = np.array(sorted(ID - IN))
subj_list = [str(id) for id in ID.tolist()]

ln = len(ID)

#%% Extract timeseries (ONE TIME: JHU cluster)

import os
import numpy as np
import nibabel as nib
import hcp_utils as hcp

# Which REST/TASK condition?
task = 'REST1' # REST1, REST2, WM, GAMBLING, MOTOR, LANGUAGE, SOCIAL, RELATIONAL, EMOTION

data_path = '/dcl01/smart/data/hpc900/' 
disks = ['disk1/', 'disk2/', 'disk3/', 'disk4/', 
         'disk5/', 'disk6/', 'disk7/', 'disk8/']
atlas = hcp.mmp # {‘mmp’, ‘ca_parcels’, ‘ca_network’, ‘yeo7’, ‘yeo17’}

ts = []

for i, sub_id in enumerate(ID):
    
    # there are 8 disks in HCP dataset
    if sub_id >= 100206 and sub_id <= 128026: # disk1
        disk = disks[1-1]
    elif sub_id >= 128127 and sub_id <= 154229: # disk2
        disk = disks[2-1]
    elif sub_id >= 179245 and sub_id <= 209329: # disk3
        disk = disks[3-1]
    elif sub_id >= 154431 and sub_id <= 178950: # disk4
        disk = disks[4-1]   
    elif sub_id >= 209834 and sub_id <= 371843: # disk5
        disk = disks[5-1] 
    elif sub_id >= 377451 and sub_id <= 587664: # disk6
        disk = disks[6-1]
    elif sub_id >= 588565 and sub_id <= 783462: # disk7
        disk = disks[7-1]
    elif sub_id >= 784565 and sub_id <= 996782: # disk8
        disk = disks[8-1]
    
    # reading timeseris
    if task == 'REST1' or task == 'REST2':
        img_lr = nib.load(data_path + disk + str(sub_id) + '/MNINonLinear/Results/rfMRI_' + task + '_LR/rfMRI_' + task + '_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(sub_id=sub_id))
        img_rl = nib.load(data_path + disk + str(sub_id) + '/MNINonLinear/Results/rfMRI_' + task + '_RL/rfMRI_' + task + '_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(sub_id=sub_id))
    else:
        img_lr = nib.load(data_path + disk + str(sub_id) + '/MNINonLinear/Results/tfMRI_' + task + '_LR/tfMRI_' + task + '_LR_Atlas_MSMAll.dtseries.nii'.format(sub_id=sub_id))
        img_rl = nib.load(data_path + disk + str(sub_id) + '/MNINonLinear/Results/tfMRI_' + task + '_RL/tfMRI_' + task + '_RL_Atlas_MSMAll.dtseries.nii'.format(sub_id=sub_id))
    
    # normalizing, parcelating, concatanating L and R    
    data_lr = img_lr.get_fdata()
    data_n_lr = hcp.normalize(data_lr) # fine-scale data
    data_p_lr = hcp.parcellate(data_n_lr, atlas)[:, :360] # targets: coarse-scale data
            
    data_rl = img_rl.get_fdata()
    data_n_rl = hcp.normalize(data_rl) # fine-scale data
    data_p_rl = hcp.parcellate(data_n_rl, atlas)[:, :360] # targets: coarse-scale data
            
    data_p = np.vstack((data_p_lr, data_p_rl))
        
    ts.append(data_p)
        
    print(task, sub_id)

np.save('/dcl01/smart/data/fvfarahani/HCP_Motion/ts_' + task, ts)

# for i in range(len(ts)):
#     if np.shape(ts[i])[0] != np.shape(ts[0])[0]:
#         print(i, np.shape(ts[i])[0])


#%% ===========================================================================
# Calculate Framewise Displacement (FD)
#     FD matrix for REST/TASK conditions
#     Compute WasserStein distances 
#     Create panda df
#     Pairplot
# =============================================================================

# FD for REST/TASK conditions
import pandas as pd

path = '/Volumes/Elements/HCP_Motion/'

sessions = [('_rfMRI_REST1_LR_Motion.txt', '_rfMRI_REST1_RL_Motion.txt'),
            ('_rfMRI_REST2_LR_Motion.txt', '_rfMRI_REST2_RL_Motion.txt'),
            ('_WM_LR_Motion.txt', '_WM_RL_Motion.txt'),
            ('_GAMBLING_LR_Motion.txt', '_GAMBLING_RL_Motion.txt'),
            ('_MOTOR_LR_Motion.txt', '_MOTOR_RL_Motion.txt'),
            ('_LANGUAGE_LR_Motion.txt', '_LANGUAGE_RL_Motion.txt'),
            ('_SOCIAL_LR_Motion.txt', '_SOCIAL_RL_Motion.txt'),
            ('_RELATIONAL_LR_Motion.txt', '_RELATIONAL_RL_Motion.txt'),
            ('_EMOTION_LR_Motion.txt', '_EMOTION_RL_Motion.txt')]

folders = ['RSmotion', 'RSmotion', 'WMmotion', 'GAMBLINGmotion', 'MOTORmotion', 
           'LANGUAGEmotion', 'SOCIALmotion', 'RELATIONALmotion', 'EMOTIONmotion']

DemoData = np.array(pd.read_excel('/Volumes/Elements/HCP_Motion/DemoData.xlsx'))
Demo = np.zeros((ln, 8))
for i in range(ln):
    ind = (DemoData[:,0] == ID[i])
    Demo[i,:] = DemoData[ind,:] # of available subjects

radius = 50 # radius of sphere in mm to convert degrees/radians to motion

FDMat = []
FDMat_mean = []

for s, f in zip(sessions, folders):
    
    os.chdir(path + f) #!rm .DS_Store
    
    tp = np.shape(np.array(pd.read_csv(str(ID[0]) + s[0], header=None, delim_whitespace=True)))[0] # time points
    FD = np.zeros((2*tp, ln))
    
    for i in range(ln): # loop over available subjects
        ts_lr = np.array(pd.read_csv(str(ID[i]) + s[0], header=None, delim_whitespace=True))
        ts_rl = np.array(pd.read_csv(str(ID[i]) + s[1], header=None, delim_whitespace=True))
        ts_lr = ts_lr[:, 0:6]; ts_rl = ts_rl[:, 0:6]
        # convert degrees/radians into motion in mm
        tmp_lr = ts_lr[:, 3:]; tmp_rl = ts_rl[:, 3:]        
        tmp_lr = (2*radius*np.pi/360)*tmp_lr # degree to motion
        tmp_rl = (2*radius*np.pi/360)*tmp_rl
        # tmp = radius*tmp # radians to motion
        ts_lr[:, 3:] = tmp_lr
        ts_rl[:, 3:] = tmp_rl
        dts_lr = np.diff(ts_lr, axis=0)
        dts_rl = np.diff(ts_rl, axis=0)
        fwd_lr = np.sum(abs(dts_lr), axis=1) # framewise displacement (FD)
        fwd_rl = np.sum(abs(dts_rl), axis=1)
        
        if (len(fwd_lr) == tp-1): # needs to be deleted
            FD[1:tp, i] = fwd_lr
        if (len(fwd_rl) == tp-1): # needs to be deleted
            FD[tp+1:, i] = fwd_lr

    FDMat.append(FD)
    FDMat_mean.append(np.mean(FD, axis=0))

FDMat_mean = np.array(FDMat_mean).T

# Compute WasserStein distances (with mean)
from scipy.stats import wasserstein_distance, energy_distance

WS_mean = np.zeros((ln,len(FDMat)))
for s in range(len(FDMat)): # loop over conditions
    for i in range(ln): # loop over subjects
        WS_mean[i, s] = wasserstein_distance(FDMat[s][:,i], np.mean(FDMat[s], axis=1))
        # WS_mean[i, r] = energy_distance(FDMat[:,i], np.mean(FDMat, axis=1))

df = pd.DataFrame({'ID':subj_list, 'Age':Demo[:,1], 'BMI':Demo[:,2],
                   'Income':Demo[:,3], 'Education':Demo[:,4], 'Gender':Demo[:,5],
                   'Depression':Demo[:,6], 'IQ':Demo[:,7],
                   'ALL: FD':       np.mean(FDMat_mean, axis=1), 
                   'ALL: WS':       np.mean(WS_mean, axis=1),
                   'REST1: FD':     FDMat_mean[:,0], 'REST1: WS':       WS_mean[:,0],
                   'REST2: FD':     FDMat_mean[:,1], 'REST2: WS':       WS_mean[:,1],
                   'WM: FD':        FDMat_mean[:,2], 'WM: WS':          WS_mean[:,2],
                   'GAMBLING: FD':  FDMat_mean[:,3], 'GAMBLING: WS':    WS_mean[:,3],
                   'MOTOR: FD':     FDMat_mean[:,4], 'MOTOR: WS':       WS_mean[:,4],
                   'LANGUAGE: FD':  FDMat_mean[:,5], 'LANGUAGE: WS':    WS_mean[:,5],
                   'SOCIAL: FD':    FDMat_mean[:,6], 'SOCIAL: WS':      WS_mean[:,6],
                   'RELATIONAL: FD':FDMat_mean[:,7], 'RELATIONAL: WS':  WS_mean[:,7],
                   'EMOTION: FD':   FDMat_mean[:,8], 'EMOTION: WS':     WS_mean[:,8],
                   }).set_index('ID')

# replace nan with mean
df['BMI'] = df['BMI'].fillna(int(df['BMI'].mean()))
df['Income'] = df['Income'].fillna(int(df['Income'].mean()))
df['Depression'] = df['Depression'].fillna(int(df['Depression'].mean()))
df['IQ'] = df['IQ'].fillna(int(df['IQ'].mean()))

# df.isnull().any() # --> to check if any value is NaN in a Pandas DataFrame

#%% Clustering: common labels across sessions
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained # clf: pip install k-means-constrained
import matplotlib.pyplot as plt
import hcp_utils as hcp

label = np.zeros((ln, len(FDMat))) # 9 number of conditions
data = []
centroids = []
for s in range(len(FDMat)): # loop over conditions 
    # Percentile based on each condition
    percentile = np.zeros((ln, 100))
    for i in range(ln):
        for p in range(100):
            percentile[i,p] = np.quantile(FDMat[s][:,i], (0.01 * (p+1)))
    
    # # Percentile based on ALL sessions
    # percentile = np.zeros((ln, 100))
    # for i in range(ln):
    #     temp = [FDMat[j][:,i] for j in range(len(FDMat))] # subset the list
    #     temp = np.hstack(temp) # concatenate
    #     for p in range(100):
    #         percentile[i,p] = np.quantile(temp, (0.01 * (p+1)))        

    data.append(percentile)
    # data = hcp.normalize(percentile.T).T  # normalization does not make sense     
    
    # K-means clustering
    n_clusters = 2
    ratio = 0.2
    size_min=int(ratio*ln) # 250
    size_max=700 # 394
    #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data[s])
    kmeans = KMeansConstrained(n_clusters=n_clusters, random_state=0,
                               size_min=size_min, size_max=size_max).fit(data[s])
    label[:,s] = kmeans.labels_
    centroids.append(kmeans.cluster_centers_)
    
    print(s)
        
    if n_clusters == 4:
        print(len(np.where(label[:,s] == 0)[0]),
              len(np.where(label[:,s] == 1)[0]),
              len(np.where(label[:,s] == 2)[0]),
              len(np.where(label[:,s] == 3)[0]))
        # REST1:      [0] -> 154 16  616 1
        # REST2:      [1] -> 55  536 6   190
        # WM:         [2] -> 553 31  4   199
        # GAMBLING:   [3] -> 215 5   21  546
        # MOTOR:      [4] -> 539 33  209 6
        # LANGUAGE:   [5] -> 230 476 5   76
        # SOCIAL:     [6] -> 136 1   632 18
        # RELATIONAL: [7] -> 721 3   54  9
        # EMOTION:    [8] -> 207 29  1   550

    if n_clusters == 3:
        print(len(np.where(label[:,s] == 0)[0]),
              len(np.where(label[:,s] == 1)[0]),
              len(np.where(label[:,s] == 2)[0]))
        # REST1:      [0] -> 695 1   91
        # REST2:      [1] -> 143 638 6
        # WM:         [2] -> 741 42  4
        # GAMBLING:   [3] -> 761 4   22
        # MOTOR:      [4] -> 734 6   47
        # LANGUAGE:   [5] -> 665 117 5
        # SOCIAL:     [6] -> 749 37  1
        # RELATIONAL: [7] -> 747 3   37
        # EMOTION:    [8] -> 746 40  1
        
    if n_clusters == 2:
        print(len(np.where(label[:,s] == 0)[0]),
              len(np.where(label[:,s] == 1)[0]))
        # REST1:      [0] -> 756 31
        # REST2:      [1] -> 757 30
        # WM:         [2] -> 764 23
        # GAMBLING:   [3] -> 774 13
        # MOTOR:      [4] -> 749 38
        # LANGUAGE:   [5] -> 781 6
        # SOCIAL:     [6] -> 786 1
        # RELATIONAL: [7] -> 774 13
        # EMOTION:    [8] -> 748 39
      
        
if n_clusters == 4: # if we have 4 clusters 
    c1 = [2, 1, 0, 3, 0, 1, 2, 0, 3] # low motion
    c2 = [0, 3, 3, 0, 2, 0, 0, 2, 0] # high motion
    label_idx = []
    label_idx.append((np.where(label[:,0] == c1[0])[0], np.where(label[:,0] == c2[0])[0])) # REST1
    label_idx.append((np.where(label[:,1] == c1[1])[0], np.where(label[:,1] == c2[1])[0])) # REST2
    label_idx.append((np.where(label[:,2] == c1[2])[0], np.where(label[:,2] == c2[2])[0])) # WM
    label_idx.append((np.where(label[:,3] == c1[3])[0], np.where(label[:,3] == c2[3])[0])) # GAMBLING
    label_idx.append((np.where(label[:,4] == c1[4])[0], np.where(label[:,4] == c2[4])[0])) # MOTOR
    label_idx.append((np.where(label[:,5] == c1[5])[0], np.where(label[:,5] == c2[5])[0])) # LANGUAGE
    label_idx.append((np.where(label[:,6] == c1[6])[0], np.where(label[:,6] == c2[6])[0])) # SOCIAL
    label_idx.append((np.where(label[:,7] == c1[7])[0], np.where(label[:,7] == c2[7])[0])) # RELATIONAL
    label_idx.append((np.where(label[:,8] == c1[8])[0], np.where(label[:,8] == c2[8])[0])) # EMOTION

if n_clusters == 3: # if we have 3 clusters 
    c1 = [0, 1, 0, 0, 0, 0, 0, 0, 0] # low motion
    c2 = [2, 0, 1, 2, 2, 1, 1, 2, 1] # high motion
    label_idx = []
    label_idx.append((np.where(label[:,0] == c1[0])[0], np.where(label[:,0] == c2[0])[0])) # REST1
    label_idx.append((np.where(label[:,1] == c1[1])[0], np.where(label[:,1] == c2[1])[0])) # REST2
    label_idx.append((np.where(label[:,2] == c1[2])[0], np.where(label[:,2] == c2[2])[0])) # WM
    label_idx.append((np.where(label[:,3] == c1[3])[0], np.where(label[:,3] == c2[3])[0])) # GAMBLING
    label_idx.append((np.where(label[:,4] == c1[4])[0], np.where(label[:,4] == c2[4])[0])) # MOTOR
    label_idx.append((np.where(label[:,5] == c1[5])[0], np.where(label[:,5] == c2[5])[0])) # LANGUAGE
    label_idx.append((np.where(label[:,6] == c1[6])[0], np.where(label[:,6] == c2[6])[0])) # SOCIAL
    label_idx.append((np.where(label[:,7] == c1[7])[0], np.where(label[:,7] == c2[7])[0])) # RELATIONAL
    label_idx.append((np.where(label[:,8] == c1[8])[0], np.where(label[:,8] == c2[8])[0])) # EMOTION

if n_clusters == 2: # if we have 2 clusters 
    c1 = [0, 0, 0, 0, 0, 0, 0, 0, 1] # low motion
    c2 = [1, 1, 1, 1, 1, 1, 1, 1, 0] # high motion
    label_idx = []
    label_idx.append((np.where(label[:,0] == c1[0])[0], np.where(label[:,0] == c2[0])[0])) # REST1
    label_idx.append((np.where(label[:,1] == c1[1])[0], np.where(label[:,1] == c2[1])[0])) # REST2
    label_idx.append((np.where(label[:,2] == c1[2])[0], np.where(label[:,2] == c2[2])[0])) # WM
    label_idx.append((np.where(label[:,3] == c1[3])[0], np.where(label[:,3] == c2[3])[0])) # GAMBLING
    label_idx.append((np.where(label[:,4] == c1[4])[0], np.where(label[:,4] == c2[4])[0])) # MOTOR
    label_idx.append((np.where(label[:,5] == c1[5])[0], np.where(label[:,5] == c2[5])[0])) # LANGUAGE
    label_idx.append((np.where(label[:,6] == c1[6])[0], np.where(label[:,6] == c2[6])[0])) # SOCIAL
    label_idx.append((np.where(label[:,7] == c1[7])[0], np.where(label[:,7] == c2[7])[0])) # RELATIONAL
    label_idx.append((np.where(label[:,8] == c1[8])[0], np.where(label[:,8] == c2[8])[0])) # EMOTION

#%%
# Visulaization (TSNE) for REST1 clustering
tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=0)
data_tsne = tsne.fit_transform(data[0])
# Getting unique labels and centroids
#label[np.where(label[:,0]==3),0] = 1 # merge a cluster (C4) with one sbj to the other outlier cluster (C3 or None)
u_labels = np.unique(label[:,0])
# plotting the results:
c = ['#0080FE', '#BF0A30'] # c = ['#BF0A30', '#FFC317', '#0080FE']
fig, ax = plt.subplots(figsize=(6.5, 4)) # 6 * 4
for i in u_labels:
    scatter = plt.scatter(data_tsne[label[:,0] == i, 0], data_tsne[label[:,0] == i, 1], 
                          label = i, c=c[int(i)], alpha=0.7)
#plt.scatter(centroids[:,0], centroids[:,1], s = 80, color = 'k')

# Hide the right and top spines
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
plt.title('T-SNE projection') #x label
plt.xlabel('Component 1') 
plt.ylabel('Component 2') 

#cmap=ListedColormap(['#C21E56', '#0437F2', '#FFC000']

# Change order of legend labels: get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
labels = ['1', '2'] # labels = ['2', '3', '1']
order = [0,1] # order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], title="Cluster") 

#plt.legend()
plt.tight_layout()
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/tSNE.pdf', bbox_inches='tight') 
plt.show() 

#%% community density, consensus, entropy --> scatterplot
from matplotlib.colors import ListedColormap
from scipy.stats import entropy
import seaborn as sns

partition = np.zeros((ln, len(FDMat))) # clustering labels across conditions
density = np.zeros((ln, 2)) # 0:high-motion; 1:low-motion; 2:neither
for i in range(ln):
    for j in range(len(FDMat)):
        if i in label_idx[j][0]: # C1: 0: low motion
            density[i, 0] += 1
        elif i in label_idx[j][1]: # C2: 1: high motion
            density[i, 1] += 1
            partition[i, j] = 1
# neither high/low motion subjects            
#density[:,2] = 9-density[:,0]-density[:,1]
# calculate entropy
ent = entropy(density.T, base=2)
# calculate consensus
consensus = np.argmax(density, axis=1) # OR (next line)
#consensus = np.zeros((ln,)); consensus[np.where(density[:,1]>=4)[0]] = 1

# Create  colormap using matplotlib
import matplotlib.colors as mcolors
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).   """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
c = mcolors.ColorConverter().to_rgb
# ONLY CHANGE THIS PART
# https://colordesigner.io/gradient-generator
my_cmap = make_colormap(
    [c('#00007F'), c('#0080FE'), 0.05, 
     c('#0080FE'), c('#00BBFE'), 0.40,
     c('#00BBFE'), c('white'), 0.50,
     c('white'), c('#FF3730'), 0.60,
     c('#FF3730'), c('#BF0A30'), 0.95,
     c('#BF0A30'), c('#590A36')])

fig, axs = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
vmin=-.5; vmax=1.5
# Conditions Partitioning
sns.heatmap(partition[3:13,:], annot=False, linewidth=1, linecolor='black', clip_on=False, 
            square=True, fmt=".2f", ax=axs[0], cmap=my_cmap, cbar=False, annot_kws={"size":7},
            yticklabels=False, vmin=vmin, vmax=vmax)
axs[0].set_xticklabels(['R1', 'R2', 'WM', 'GB', 'MT', 'LG', 'SC', 'RL', 'EM'], rotation=0)
axs[0].set_ylabel('Participants', rotation=90)
axs[0].xaxis.tick_top()
axs[0].set_title('Clustering acrosss conditions', y=1.1)
# Density
temp1 = np.where(density >= 5, 1, 0); temp2=np.zeros((787,2))
temp2[:,0] = np.where(temp1[:,0] == 1, 0, 0.5)
temp2[:,1] = np.where(temp1[:,1] == 1, 1, 0.5)
text = np.array([['9', '0'], ['8', '1'], ['4', '5'], ['9', '0'], ['9', '0'], 
                 ['4', '5'], ['7', '2'], ['9', '0'], ['2', '7'], ['8', '1']])
sns.heatmap(temp2[3:13,:], annot=text, linewidth=1, linecolor='black', clip_on=False, 
            square=True, fmt='', ax=axs[1], cmap=my_cmap, cbar=False, annot_kws={"size":12},
            yticklabels=False, vmin=vmin, vmax=vmax)
axs[1].set_xticklabels(['C1', 'C2'], rotation=0)
axs[1].xaxis.tick_top()
axs[1].set_title('Density', y=1.1)
# Consensus
sns.heatmap(pd.DataFrame(consensus[3:13]), annot=False, linewidth=1, linecolor='black', clip_on=False, 
            square=True, fmt=".2f", ax=axs[2], cmap=my_cmap, cbar=False,
            xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax)
axs[2].set_title('Consensus\n clustering', y=1.06)
# Entropy
sns.heatmap(pd.DataFrame(ent[3:13]), annot=False, linewidth=1, linecolor='black', clip_on=False, 
            square=True, fmt=".2f", ax=axs[3], cmap='binary', cbar=False,
            xticklabels=False, yticklabels=False)
axs[3].set_title('Entropy', y=1.1)

fig.tight_layout()
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/Consensus.pdf', bbox_inches='tight') 

#%% plot percentile
fig = plt.subplots(figsize=(15, 8))
ax = sns.heatmap(percentile, square=False, cmap='jet', cbar=True,
            yticklabels=False, vmin=0, vmax=0.3, cbar_kws={"shrink":.7, "pad":.03})

# change the fontsize of cbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

plt.ylabel('Subjects (#787)', rotation=90, size=14)
plt.title('Framewise Displacement (FD) Percentiles', size=14)

plt.tight_layout()
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/percentile.pdf', bbox_inches='tight') 


#%% scatterplot 
x, y = df['ALL: FD'], df['BMI']
c = consensus
s = 4 * (df['IQ']-df['IQ'].min()) # np.random.randint(10, 220, size=N)

fig, ax = plt.subplots(figsize=(6,4))
scatter = ax.scatter(x, y, c=c, s=s, alpha=0.7, cmap=ListedColormap(['#0080FE', '#BF0A30'])) # ['#0080FE', '#BF0A30', '#FFC000']

plt.xlabel('Framewise displacement (mean)', fontsize=12) #x label
plt.ylabel('BMI', fontsize=12) #y label
plt.xlim() # plt.xlim(right=0.5)

# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.7, num=4)
legend1 = ax.legend(handles, ['5', '10', '15', '20'], loc='lower right', title="Gf", frameon=True)
ax.add_artist(legend1)

# produce a legend with the unique colors from the scatter
handles, labels = scatter.legend_elements()
legend2 = ax.legend(handles, ['C1: Low motion', 'C2: High motion'] , loc='upper right', frameon=True) # title="Motion",

# Hide the right and top spines
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.tight_layout()
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/Scatterplot.pdf', bbox_inches='tight') 
plt.show()

#%% plot overlapping heatmap and venn diagram
import seaborn as sns

ticks = ['R1', 'R2', 'WM', 'GB', 'MT', 'LG', 'SC', 'RL', 'EM']
sns.set(font_scale=1)
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
plt.subplots_adjust(wspace=0.1)
clusters = ['C1: Low Motion', 'C2: High Motion']

for c, ax in zip(range(2), axs.ravel()): # 0 (low motion) or 1 (high motion)

    heatmap = np.ones((len(FDMat), len(FDMat)))
    for i in range(1, len(FDMat)):
        j = 0
        while j<i:
            intersect = list(set(label_idx[i][c]) & set(label_idx[j][c]))
            heatmap[i,j] = len(intersect)/len(set(label_idx[i][c]))
            heatmap[j,i] = len(intersect)/len(set(label_idx[j][c]))
            j += 1
    
    ax = sns.heatmap(heatmap, annot=True, linewidth=1, square=True, fmt=".2f", ax=ax, vmax=1, vmin=0.4,
                cmap='RdBu', xticklabels=ticks, yticklabels=ticks, annot_kws={"size":7}, # sns.cubehelix_palette(as_cmap=True) # Spectral
                cbar_kws=dict(location="right", pad=0.02, shrink=0.5))
    
    # change the fontsize of cbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    #cbar.set_ticks(np.arange(0.4, 1.09, 0.1))
    
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=8)
    plt.xticks(rotation=0) 
    ax.set_title(clusters[c], y=1.1)

# save the graph
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/cluster_overlap.pdf', bbox_inches='tight') 

# =============================================================================
# # Venn digram
# from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
# palette = sns.color_palette("Set2", 12)
# c = 0; i = 5; j = 2
# a, b = len(set(label_idx[i][c])), len(set(label_idx[j][c]))
# ab = len(list(set(label_idx[i][c]) & set(label_idx[j][c])))
# v=venn2(subsets = (a-ab, b-ab, ab), set_labels = ('Language', 'WM'),
#         set_colors=(palette[0], palette[1]), alpha = 1)
# c=venn2_circles(subsets = (a-ab, b-ab, ab), linestyle='dashed', linewidth=1, color="k")
# plt.show()
# =============================================================================

#%% Displot with multiple distributions

import numpy as np
import pandas as pd
import seaborn as sns

condition = 'REST1: FD'
measure = 'BMI' # REST1: FD, BMI, Age, BMI, Income, Education, Gender

df['Motion Cluster: REST1'] = label[:,0]
df['Motion Cluster: REST2'] = label[:,1]
df['Motion Cluster: WM'] = label[:,2]
df['Motion Cluster: GAMBLING'] = label[:,3]
df['Motion Cluster: MOTOR'] = label[:,4]
df['Motion Cluster: LANGUAGE'] = label[:,5]
df['Motion Cluster: SOCIAL'] = label[:,6]
df['Motion Cluster: RELATIONAL'] = label[:,7]
df['Motion Cluster: EMOTION'] = label[:,8]

df['Motion Group'] = 0 
df.loc[df['ALL: FD'] >= df['ALL: FD'].mean(), 'Motion Group'] = 1

chart = sns.displot(data=df, x=measure, hue='Motion Cluster: REST1', hue_order = range(n_clusters), kind='kde', fill=True, palette=sns.color_palette('bright')[:4], height=5, aspect=1.5)

## Legend title
chart._legend.set_title('Motion Categories')

# Replacing labels
# new_labels = ['Q1', 'Q2', 'Q3', 'Q4']
# for t, l in zip(chart._legend.texts, new_labels):
#     t.set_text(l)


#%% Centroids and percentiles ditribution

#label[label == 1] = 0
#df['Motion Cluster'] = label

c1 = 1; c2 = 0 

label_c1 = np.where(label == c1)[0]
label_c2 = np.where(label == c2)[0]

# dataframe
new_df = pd.DataFrame()
new_df['Centroid (k-means)'] = np.concatenate((centroids[0][c1,:-1], centroids[0][c2,:-1]), axis=0)

# # Percentiles ditribution
# percentile_c1 = np.zeros((9, 100))
# percentile_c2 = np.zeros((9, 100))
# for s in range(len(FDMat)): # sessions
#     for p in range(100):
#         percentile_c1[s,p] = np.quantile(FDMat[s][:,label_c1].flatten(), (0.01 * (p+1)))
#         percentile_c2[s,p] = np.quantile(FDMat[s][:,label_c2].flatten(), (0.01 * (p+1)))

# new_df['% Rest 1'] = np.concatenate((percentile_c1[0,:-1], percentile_c2[0,:-1]), axis=0)
# new_df['% Rest 2'] = np.concatenate((percentile_c1[1,:-1], percentile_c2[1,:-1]), axis=0)
# new_df['% WM'] = np.concatenate((percentile_c1[2,:-1], percentile_c2[2,:-1]), axis=0)
# new_df['% Gambling'] = np.concatenate((percentile_c1[3,:-1], percentile_c2[3,:-1]), axis=0)
# new_df['% Motor'] = np.concatenate((percentile_c1[4,:-1], percentile_c2[4,:-1]), axis=0)
# new_df['% Language'] = np.concatenate((percentile_c1[5,:-1], percentile_c2[5,:-1]), axis=0)
# new_df['% Social'] = np.concatenate((percentile_c1[6,:-1], percentile_c2[6,:-1]), axis=0)
# new_df['% Relational'] = np.concatenate((percentile_c1[7,:-1], percentile_c2[7,:-1]), axis=0)
# new_df['% Emotion'] = np.concatenate((percentile_c1[8,:-1], percentile_c2[8,:-1]), axis=0)

new_df['Label'] = np.concatenate((np.repeat(['C1: Low'], 99),
                                       np.repeat(['C2: High'], 99))) # repeat or tile 

chart = sns.displot(data=new_df, x='Centroid (k-means)', hue='Label', kind='kde', 
                    legend=False, fill=True, palette=['#BF0A30', '#0080FE'], height=4, aspect=1.7)
## Legend title
plt.legend(labels=["C1: Low motion","C2: High motion"], loc = "upper right", # title='Motion'
           fontsize = 10) # title_fontsize = "8"

plt.tight_layout()
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/distplot.pdf', bbox_inches='tight') 
plt.show()

#%% Radar chart
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
# Set data (train: REST1, test on others)
c1 = 0; c2 = 1
temp = df.groupby('Motion Cluster: REST1')
df_radar = pd.DataFrame({
'Group':      ['A','B'],
'R1':     [temp['REST1: FD'].mean()[c1], temp['REST1: FD'].mean()[c2]],
'R2':     [temp['REST2: FD'].mean()[c1], temp['REST2: FD'].mean()[c2]],
'WM':         [temp['WM: FD'].mean()[c1], temp['WM: FD'].mean()[c2]],
'GB':   [temp['GAMBLING: FD'].mean()[c1], temp['GAMBLING: FD'].mean()[c2]],
'MT':      [temp['MOTOR: FD'].mean()[c1], temp['MOTOR: FD'].mean()[c2]],
'LG':   [temp['LANGUAGE: FD'].mean()[c1], temp['LANGUAGE: FD'].mean()[c2]],
'SC':     [temp['SOCIAL: FD'].mean()[c1], temp['SOCIAL: FD'].mean()[c2]],
'RL': [temp['RELATIONAL: FD'].mean()[c1], temp['RELATIONAL: FD'].mean()[c2]],
'EM':    [temp['EMOTION: FD'].mean()[c1], temp['EMOTION: FD'].mean()[c2]],
})

# =============================================================================
# # Set data (train condition set separately)
# #temp = df.groupby('Motion Cluster')
# df_radar = pd.DataFrame({
# 'Group':      ['A','B'],
# 'Rest 1':     [df.groupby('Motion Cluster: REST1')['REST1: FD'].mean()[c1[0]], df.groupby('Motion Cluster: REST1')['REST1: FD'].mean()[c2[0]]],
# 'Rest 2':     [df.groupby('Motion Cluster: REST2')['REST2: FD'].mean()[c1[1]], df.groupby('Motion Cluster: REST2')['REST2: FD'].mean()[c2[1]]],
# 'WM':         [df.groupby('Motion Cluster: WM')['WM: FD'].mean()[c1[2]], df.groupby('Motion Cluster: WM')['WM: FD'].mean()[c2[2]]],
# 'Gambling':   [df.groupby('Motion Cluster: GAMBLING')['GAMBLING: FD'].mean()[c1[3]], df.groupby('Motion Cluster: GAMBLING')['GAMBLING: FD'].mean()[c2[3]]],
# 'Motor':      [df.groupby('Motion Cluster: MOTOR')['MOTOR: FD'].mean()[c1[4]], df.groupby('Motion Cluster: MOTOR')['MOTOR: FD'].mean()[c2[4]]],
# 'Language':   [df.groupby('Motion Cluster: LANGUAGE')['LANGUAGE: FD'].mean()[c1[5]], df.groupby('Motion Cluster: LANGUAGE')['LANGUAGE: FD'].mean()[c2[5]]],
# 'Social':     [df.groupby('Motion Cluster: SOCIAL')['SOCIAL: FD'].mean()[c1[6]], df.groupby('Motion Cluster: SOCIAL')['SOCIAL: FD'].mean()[c2[6]]],
# 'Relational': [df.groupby('Motion Cluster: RELATIONAL')['RELATIONAL: FD'].mean()[c1[7]], df.groupby('Motion Cluster: RELATIONAL')['RELATIONAL: FD'].mean()[c2[7]]],
# 'Emotion':    [df.groupby('Motion Cluster: EMOTION')['EMOTION: FD'].mean()[c1[8]], df.groupby('Motion Cluster: EMOTION')['EMOTION: FD'].mean()[c2[8]]],
# })
# =============================================================================

# ------- PART 1: Create background
 
# number of variable
categories=list(df_radar)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([.05,.10,.15,.20,.25], ["0.05","0.10","0.15","0.20","0.25"], color="grey", size=7)
plt.ylim(0,.28)
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df_radar.loc[0].drop('Group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="C1: Low motion")
ax.fill(angles, values, '#0080FE', alpha=0.15)  
 
# Ind2
values=df_radar.loc[1].drop('Group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="C2: High motion")
ax.fill(angles, values, '#BF0A30', alpha=0.15)
 
# Add legend
plt.legend(loc='lower center', bbox_to_anchor=(.5, -.26)) # bbox_to_anchor=(0.1, 0.1)

# Show the graph
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/Radar.pdf', bbox_inches='tight') 
plt.show()

#%% Demographic comparision (Two-Sample T-Test)
import scipy.stats as stats

c1 = 0; c2 = 1 # 0: low, 1: high
temp = df.groupby('Motion Cluster: REST1')
df_demo = pd.DataFrame({
'Group':     ['A','B'],
'Age':       [temp['Age'].mean()[c1], temp['Age'].mean()[c2]],
'BMI':       [temp['BMI'].mean()[c1], temp['BMI'].mean()[c2]],
'Income':    [temp['Income'].mean()[c1], temp['Income'].mean()[c2]],
'Education': [temp['Education'].mean()[c1], temp['Education'].mean()[c2]],
'Depression': [temp['Depression'].mean()[c1], temp['Depression'].mean()[c2]],
'IQ': [temp['IQ'].mean()[c1], temp['IQ'].mean()[c2]],
})

"""
Before conducting the two-sample T-Test we need to find if the given 
data groups have the same variance. If the ratio of the larger data groups 
to the small data group is less than 4:1 then we can consider that the 
given data groups have equal variance. To find the variance of a data group, 
we can use the below syntax,
"""
# Print the variance of both data groups
measure = 'IQ' # Age, BMI, Income, Education, Depression, IQ
data_group1 = df[df['Consensus'] == c1][measure]
data_group2 = df[df['Consensus'] == c2][measure]
print('C1 --> Population: ', data_group1.count(), 
      '\tMean:', data_group1.mean(), '\tSD:', data_group1.std()) 
print('C2 --> Population: ', data_group2.count(), 
      '\tMean:', data_group2.mean(), '\tSD:', data_group2.std()) 
print('Ration: ', data_group1.var()/data_group2.var())

# define permutation test using monte-carlo method
def perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

# Perform permutation test
print('P-value for', measure, '(permutation test):', 
      perm_test(data_group1, data_group2, 100000))

# Perform the two sample t-test with equal variances
print('P-value for', measure, '(two sample t-test):', 
      stats.ttest_ind(a=data_group1, b=data_group2, equal_var=True)[1])

# False Discovery Rate
# https://tools.carbocation.com/FDR

#%% BOXPLOT: Display every observations over the boxplot

c1 = 0; c2 = 1 # 0: low, 1: high

dfA1 = df.loc[df['Motion Cluster: REST1'] == c1][['REST1: FD', 'REST2: FD', 'WM: FD', 'GAMBLING: FD',
                                          'MOTOR: FD', 'LANGUAGE: FD', 'SOCIAL: FD', 'RELATIONAL: FD', 'EMOTION: FD']]
dfA2 = df.loc[df['Motion Cluster: REST1'] == c2][['REST1: FD', 'REST2: FD', 'WM: FD', 'GAMBLING: FD',
                                          'MOTOR: FD', 'LANGUAGE: FD', 'SOCIAL: FD', 'RELATIONAL: FD', 'EMOTION: FD']]

ticks = ['R1', 'R2', 'WM', 'GB', 'MT', 
         'LG', 'SC', 'RL', 'EM']

dfA1.columns = ticks; dfA2.columns = ticks # rname columns

names = []
valsA1, xsA1, valsA2, xsA2 = [],[], [],[]

for i, col in enumerate(dfA1.columns):
    valsA1.append(dfA1[col].values)
    valsA2.append(dfA2[col].values)
    names.append(col)
    # Add some random "jitter" to the data points
    xsA1.append(np.random.normal(i*3-0.5, 0.075, dfA1[col].values.shape[0]))
    xsA2.append(np.random.normal(i*3+0.5, 0.075, dfA2[col].values.shape[0]))

fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(11, 5))

bpA1 = ax1.boxplot(valsA1, labels=names, positions=np.array(range(len(dfA1.T)))*3-0.5, sym='', widths=0.6)
bpA2 = ax1.boxplot(valsA2, labels=names, positions=np.array(range(len(dfA2.T)))*3+0.5, sym='', widths=0.6)
# Optional: change the color of 'boxes', 'whiskers', 'caps', 'medians', and 'fliers'
plt.setp(bpA1['medians'], linewidth=2, linestyle='-', color='#800020')
plt.setp(bpA2['medians'], linewidth=2, linestyle='-', color='#800020')

palette=['#0080FE', '#BF0A30']
size = 10 # scatter point size
for xA1, xA2, valA1, valA2 in zip(xsA1, xsA2, valsA1, valsA2):
    ax1.scatter(xA1, valA1, alpha=0.4, color=palette[0], s=size, linewidth=0.2, edgecolors='k') # plt.plot(xA1, valA1, 'r.', alpha=0.4)
    ax1.scatter(xA2, valA2, alpha=0.4, color=palette[1], s=size, linewidth=0.2, edgecolors='k')
    
# Use the pyplot interface to customize any subplot...
# First subplot
plt.sca(ax1)
plt.xticks(range(0, len(ticks) * 3, 3), ticks, fontsize=12)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylim(0.05, 0.6)
plt.ylabel("Framewise Displacement", fontweight='normal', fontsize=12)
# plt.xlabel("Conditions", fontweight='normal', fontsize=10)
plt.plot([], c=palette[0], label='C1: Low motion', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c=palette[1], label='C2: High motion', marker='o', linestyle='None', markersize=8)
# =============================================================================
# # Statistical annotation
# xs1 = np.array([-0.5]) # +3
# xs2 = np.array([0.5])
# for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
#     y, h, col = max(datasetA1[:,int((x1+x2)/6)].max(), datasetA2[:,int((x1+x2)/6)].max()) + 0.4, 0.12, 'k'
#     plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#     plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)      
# # Create empty plot with blank marker containing the extra label
# #plt.text(20.81, 5.18, "*", ha='center', va='bottom', color=col, size=14, zorder=10) 
# #plt.plot([], [], " ", label='Significant Mean ($P\leq 0.05$)', color='black')    
# #plt.legend(prop={'size':16})  
# =============================================================================

# Unified legend  
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 12})  
    
sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)

# Adjust the layout of the plot
plt.tight_layout()
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/Boxplot.pdf', bbox_inches='tight')

plt.show()
    
#%% Overlapping densities (Ridgeline Plots): Distribution of FDs across all rest/task conditions

# preparing data (average FD)
data = np.concatenate((df['REST1: FD'], df['WM: FD'], df['GAMBLING: FD'], df['MOTOR: FD'],
                     df['LANGUAGE: FD'], df['SOCIAL: FD'], df['RELATIONAL: FD'], df['EMOTION: FD']))
df_full = pd.DataFrame(data=data, columns=['FD'])
df_full['Condition'] = np.repeat(['REST1', 'WM', 'GAMBLING', 'MOTOR',
                   'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION'], ln) # repeat or tile 
# preparing data (FD for all time-points)
# =============================================================================
# data = np.concatenate((FDMat[0].flatten(), FDMat[2].flatten(), FDMat[3].flatten(), FDMat[4].flatten(),
#                        FDMat[5].flatten(), FDMat[6].flatten(), FDMat[7].flatten(), FDMat[8].flatten()))
# df_full = pd.DataFrame(data=data, columns=['FD'])
# df_full['Condition'] = np.concatenate((np.repeat(['REST1'], ln*len(FDMat[0])),
#                                        np.repeat(['WM'], ln*len(FDMat[2])),
#                                        np.repeat(['GAMBLING'], ln*len(FDMat[3])),
#                                        np.repeat(['MOTOR'], ln*len(FDMat[4])),
#                                        np.repeat(['LANGUAGE'], ln*len(FDMat[5])),
#                                        np.repeat(['SOCIAL'], ln*len(FDMat[6])),
#                                        np.repeat(['RELATIONAL'], ln*len(FDMat[7])),
#                                        np.repeat(['EMOTION'], ln*len(FDMat[8])))) # repeat or tile 
# =============================================================================
# remove outliers
df_full['FD'][df_full['FD'] > 0.75] = df_full['FD'].mean()


# preparing data based on percentile
# =============================================================================
# data = np.concatenate((new_df['% Rest 1'], new_df['% Rest 2'], new_df['% WM'], new_df['% Gambling'], new_df['% Motor'],
#                      new_df['% Language'], new_df['% Social'], new_df['% Relational'], new_df['% Emotion']))
# df_full = pd.DataFrame(data=data, columns=['FD'])
# df_full['Condition'] = np.repeat(['REST1', 'REST2', 'WM', 'GAMBLING', 'MOTOR',
#                    'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION'], 99*2) # repeat or tile 
# df_full['Label'] = np.tile(new_df['Label'], 9) # repeat or tile 
# =============================================================================

# Ridge plots
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
palette = sns.color_palette("Set2", 12)
g = sns.FacetGrid(df_full, palette=palette, row="Condition", hue="Condition", aspect=9, height=1.2)
g.map_dataframe(sns.kdeplot, x="FD", fill=True, alpha=.7)
g.map_dataframe(sns.kdeplot, x="FD", color='black')
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, color='black', fontsize=13,
            ha="left", va="center", transform=ax.transAxes)
    
g.map(label, "Condition")
g.fig.subplots_adjust(hspace=-.5)
g.set_titles("")
g.set(yticks=[], xlabel="FD")
g.despine(left=True)
plt.suptitle('Framewise Displacement by Condition', y=0.98)

# plt.savefig('/Volumes/Elements/HCP_Motion/FD_dist_all.pdf') 

#%% Pairplot
import seaborn as sns
import matplotlib.pyplot as plt

path = '/Volumes/Elements/HCP_Motion/'

df['Consensus'] = consensus

# motion cluster label?
idx_low = np.where(consensus == 0)[0] # Motion -> 0: Low, 1: High
idx_low = np.sort(np.random.choice(idx_low, size=sum(consensus), replace = False)) # make equal size
idx_high = np.where(consensus == 1)[0] # Motion -> 0: Low, 1: High
idx_high = np.sort(np.random.choice(idx_high, size=sum(consensus), replace = False)) # make equal size
idx = np.concatenate((idx_low, idx_high), axis=0)

g = sns.pairplot(df.iloc[idx][['Age', 'BMI', 'Depression', 'IQ', 'Consensus']], 
             hue='Consensus', palette=['#0080FE', '#BF0A30'], 
             kind='reg', 
             plot_kws={
                 #'line_kws':{'color':'red'}, 
                 'scatter_kws': {'alpha': 0.1, 's': 30}
                 })

# place legend 
sns.move_legend(g, "center right", frameon=True)

plt.tight_layout()
#plt.savefig(path + '/Figures/Pairplot.pdf') 
plt.show()

#%% Pairplot
import seaborn as sns
import matplotlib.pyplot as plt

path = '/Volumes/Elements/HCP_Motion/'

# Age
df['Age_class'] = np.where(df['Age'] > df['Age'].mean(), 'Above AVG', 'Below AVG')
sns.pairplot(df[['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL', 'Age_class']], hue='Age_class') # palette ='coolwarm'
plt.savefig(path + '/Figures_REST/Age.pdf') 

sns.jointplot(x ='Age', y ='REST1_LR', data=df, kind ='kde')
plt.savefig(path + '/Figures_REST/Age_joint.pdf') 
# KDE shows the density where the points match up the most
# Here we can see 'REST1_LR' on the y axis and 'Age' on the x axis as well as a linear relationship between the two that suggests that the Age increases with the FD distiance.

# BMI
df['BMI_class'] = np.where(df['BMI'] > df['BMI'].mean(), 'High', 'Low')
sns.pairplot(df[['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL', 'BMI_class']], hue='BMI_class')
plt.savefig(path + '/Figures_REST/BMI.pdf') 

sns.jointplot(x ='BMI', y ='REST1_LR', data=df, kind ='kde')
plt.savefig(path + '/Figures_REST/BMI_joint.pdf') 

# Income
df['Income_class'] = np.where(df['Income'] > df['Income'].mean(), 'High', 'Low')
sns.pairplot(df[['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL', 'Income_class']], hue='Income_class')
plt.savefig(path + '/Figures_REST/Income.pdf') 

sns.jointplot(x ='Income', y ='REST1_LR', data=df, kind ='kde')
plt.savefig(path + '/Figures_REST/Income_joint.pdf') 

# Education
df['Education_class'] = np.where(df['Education'] > df['Education'].mean(), 'High', 'Low')
sns.pairplot(df[['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL', 'Education_class']], hue='Education_class')
plt.savefig(path + '/Figures_REST/Education.pdf') 

sns.jointplot(x ='Education', y ='REST1_LR', data=df, kind ='kde')
plt.savefig(path + '/Figures_REST/Education_joint.pdf') 

# Gender
sns.pairplot(df[['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL', 'Gender']], hue='Gender')
plt.savefig(path + '/Figures_REST/Gender.pdf') 

sns.jointplot(x ='Gender', y ='REST1_LR', data=df, kind ='kde')
plt.savefig(path + '/Figures_REST/Gender_joint.pdf') 


#%% ===========================================================================
# Identification accuracies across FCs
#     1. Define desired REST/TASK conditions
#     2. Stratifying subjects based on demographic data or motion (optional)
#     3. Calculate FC (correlation) + fisher r-to-z
#     4. Construct SIMILARITY matrix among pairs
#     5. Calculate identification accuracies + save acc + plot
# =============================================================================

import numpy as np
import pandas as pd 
import seaborn as sns
import hcp_utils as hcp
import matplotlib.pyplot as plt
from itertools import combinations 
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional

tasks = ['REST1', 'REST2', 'WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']

# motion cluster label?
c_label = 0 # Motion -> 0: Low, 1: High

# Stratifying subjects
#indices = np.where(Demo[:,2] <= 100000)[0] # INCLUDE ALL SUBJECTS
#indices = np.where(Demo[:,2] >= np.nanmean(Demo[:,2]))[0] # Sbj_idx: BMI >= median(BMI)
#indices = np.where(Demo[:,2] <= np.nanquantile(Demo[:,2], 0.75))[0] # Sbj_idx: BMI >= median(BMI)
#indices = np.where(Demo[:,5] == 1)[0] # Gender (0 or 1)
#indices = np.where(Demo[:,1] <= np.nanmean(Demo[:,1]))[0] # Age
#indices = np.where(Demo[:,3] >= np.nanmean(Demo[:,3]))[0] # Income
#indices = np.where(Demo[:,4] <= np.nanmean(Demo[:,4]))[0] # Education
#indices = df.index[df['ALL: WS'] >= df['ALL: WS'].mean()] # Motion
#c1 = 2; c2 = 0; indices = df.index[df['Motion Cluster: REST1'] != c2] # Motion
indices = np.where(consensus == c_label)[0] # Motion -> 0: Low, 1: High
indices = np.sort(np.random.choice(indices, size=sum(consensus), replace=False)) # make equal size
#indices = np.where(np.mean(WS_mean, axis=1) <= np.nanmean(np.mean(WS_mean, axis=1)))[0] # Motion

# accuracy matrix
acc = np.zeros((len(tasks),len(tasks)))

# Get all combinations of tasks (pairs)
comb = combinations(tasks, 2) 
  
# run across all combinations 
for c in list(comb):
    
    # extract wanted subjects based on indices (in numpy arrays)
    ts_1 = np.load('/Volumes/Elements/HCP_Motion/ts_' + c[0] + '.npy')[indices]
    ts_2 = np.load('/Volumes/Elements/HCP_Motion/ts_' + c[1] + '.npy')[indices]
    
    m = int(np.where(np.array(tasks) == c[0])[0])
    n = int(np.where(np.array(tasks) == c[1])[0])
    
    ln_sample = len(ts_1)
    
    # calculate correlation + fisher r-to-z
    # fisher r-to-z transform: to make the values of FC matrix to be of normal distribution
    corr_lr = np.arctanh(correlation_measure.fit_transform(ts_1))
    corr_rl = np.arctanh(correlation_measure.fit_transform(ts_2))
        
    # get the lower triangular
    tril_lr = []
    tril_rl = []
    for i in range(ln_sample): 
        # lower triangular
        tril_lr.append(corr_lr[i][np.tril_indices(corr_lr.shape[2], k = -1)]) # K = -1 [without diagonal] or 0 [with]
        tril_rl.append(corr_rl[i][np.tril_indices(corr_rl.shape[2], k = -1)])  
    
    tril_lr = np.array(tril_lr)
    tril_rl = np.array(tril_rl)
    
    #% SIMILARITY (i.e.,correlation coefficient between two flattened adjacency matrices) 
    sim1 = np.zeros((ln_sample,ln_sample))
    for k in range(ln_sample):
        for l in range(ln_sample):
            sim1[k,l] = np.corrcoef(tril_lr[k], tril_rl[l])[0, 1]
       
    sim2 = np.zeros((ln_sample,ln_sample))
    for k in range(ln_sample):
        for l in range(ln_sample):
            sim2[k,l] = np.corrcoef(tril_rl[k], tril_lr[l])[0, 1]        
    
    # Normalizing similarity matrix; VERY IMPORTANT
    sim1 = hcp.normalize(sim1)
    sim2 = hcp.normalize(sim2) 
        
    sim = hcp.normalize((sim1+sim2)/2)
    
    # Get index for the highest value
    index = sim.argsort()[:,-1]
    # binarize
    for k in range(ln_sample):
        sim[k,:] = [0 if i < sim[k,index[k]] else 1 for i in sim[k,:]]
    # plot        
    plt.imshow(sim)
    plt.colorbar()
    
    trace = np.trace(sim)
    accuracy = trace/ln_sample
    print(c[0], c[1], accuracy)
    
    acc[m,n] = accuracy

#np.save('/Volumes/Elements/HCP_Motion/acc_lowMOTION_KmeansRND@', acc)

#%% plot fingerprinting matrix
plt.imshow(percentile)
#plt.colorbar()

#%% Plot comparable trianglur matrices (two halves)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

feature = 'MOTION_KmeansRND' # BMI, MOTION, MOTION_KmeansConstrained, MOTION_median, MOTION_KmeansRND

tasks = ['R1', 'R2', 'WM', 'GB', 'MT', 'LG', 'SC', 'RL', 'EM']

acc_high = np.load('/Volumes/Elements/HCP_Motion/acc_high' + feature + '.npy')
acc_low = np.load('/Volumes/Elements/HCP_Motion/acc_low' + feature + '2.npy')
acc_high = acc_high + acc_high.T
acc_low = acc_low + acc_low.T

mask1 = np.triu(np.ones_like(acc_low, dtype=bool))
mask2 = np.tril(np.ones_like(acc_high, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))

cmap = 'hot_r'
sns.heatmap(acc_low, mask=mask1, cmap=cmap, vmax=1, vmin=0.4, center=0.70, fmt=".2f", 
            annot=True, annot_kws={'size':16, 'rotation':-45}, square=True, linewidths=1, cbar=False, ax=ax) # cbar_kws={"shrink":.7, "pad":.05}
sns.heatmap(acc_high, mask=mask2, cmap=cmap, vmax=1, vmin=0.4, center=0.70, fmt=".2f", 
            annot=True, annot_kws={'size':16, 'rotation':-45}, square=True, linewidths=1, cbar=False, ax=ax)

# change the fontsize of cbar
#cbar = ax.collections[0].colorbar
#cbar.ax.tick_params(labelsize=16)

# ticks
yticks = [i.upper() for i in tasks]
xticks = [i.upper() for i in tasks]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=-45)
plt.xticks(plt.xticks()[0], labels=xticks, rotation=-45)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=18)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=18)
ax.xaxis.tick_top()

# the following lines color and hatch the axes background, only the diagonals are visible
ax.patch.set_facecolor('white')
ax.patch.set_edgecolor('black')
ax.patch.set_hatch('xx')

# adjust and save the eplot
plt.tight_layout()
#plt.savefig('/Volumes/Elements/HCP_Motion/Figures/ID_' + feature + '.pdf', bbox_inches='tight')
plt.show()    

#%% Plot trianglur matrices (half)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

feature = 'MOTION' # BMI, MOTION

#acc = np.load('/Volumes/Elements/HCP_Motion/acc_' + feature + '.npy')
acc_high = np.load('/Volumes/Elements/HCP_Motion/acc_highMOTION_KmeansConstrained.npy')
acc_low = np.load('/Volumes/Elements/HCP_Motion/acc_lowMOTION_KmeansConstrained.npy')

#acc = acc + acc.T
acc_high = acc_high + acc_high.T
acc_low = acc_low + acc_low.T

tasks = ['REST1', 'REST2', 'WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']

# Plotting the lower triangular matrix for one subject

def get_lower_tri_heatmap(df, output='acc.pdf'):
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Remove diagonal elements
    mask[np.diag_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    #cmap = sns.diverging_palette(220, 10, as_cmap=True); center=0.5
    cmap = cm.get_cmap('jet'); center=0.75 # jet doesn't have white color
    #cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(df, mask=mask, cmap=cmap, vmax=1, vmin=0.5, center=center,
            annot=True, square=True, linewidths=1, cbar_kws={"shrink": .5})
    
    # ticks
    yticks = [i.upper() for i in tasks]
    xticks = [i.upper() for i in tasks]
    
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks, rotation=90)
        
    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(output)

os.chdir('/Volumes/Elements/HCP_Motion/')
#df = pd.DataFrame(acc) 
#get_lower_tri_heatmap(df, output='acc_' + feature + '.pdf') 
df = pd.DataFrame(acc_high) 
get_lower_tri_heatmap(df, output='acc_high' + feature + '.pdf')  
df = pd.DataFrame(acc_low) 
get_lower_tri_heatmap(df, output='acc_low' + feature + '.pdf')      


#%% ===========================================================================
# Connectome-based Predictive Modeling (CPM)
#     1. ...
#     2. ...
#     3. ...
#     4. ...
#     5. ...
# =============================================================================

# Define CPM functions
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
#%matplotlib inline
import pandas as pd
import seaborn as sns
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def mk_kfold_indices(subj_list, k=10):
    """
    Splits list of subjects into k folds for cross-validation.
    """
    
    n_subs = len(subj_list)
    n_subs_per_fold = n_subs//k # floor integer for n_subs_per_fold

    indices = [[fold_no]*n_subs_per_fold for fold_no in range(k)] # generate repmat list of indices
    remainder = n_subs % k # figure out how many subs are left over
    remainder_inds = list(range(remainder))
    indices = [item for sublist in indices for item in sublist]    
    [indices.append(ind) for ind in remainder_inds] # add indices for remainder subs

    assert len(indices)==n_subs, "Length of indices list does not equal number of subjects, something went wrong"

    np.random.shuffle(indices) # shuffles in place

    return np.array(indices)


def split_train_test(subj_list, indices, test_fold):
    """
    For a subj list, k-fold indices, and given fold, returns lists of train_subs and test_subs
    """

    train_inds = np.where(indices!=test_fold)
    test_inds = np.where(indices==test_fold)

    train_subs = []
    for sub in subj_list[train_inds]:
        train_subs.append(sub)

    test_subs = []
    for sub in subj_list[test_inds]:
        test_subs.append(sub)

    return (train_subs, test_subs)


def get_train_test_data(all_fc_data, train_subs, test_subs, behav_data, behav):

    """
    Extracts requested FC and behavioral data for a list of train_subs and test_subs
    """

    train_vcts = all_fc_data.loc[train_subs, :]
    test_vcts = all_fc_data.loc[test_subs, :]

    train_behav = behav_data.loc[train_subs, behav]

    return (train_vcts, train_behav, test_vcts)


def select_features(train_vcts, train_behav, r_thresh=0.2, corr_type='pearson', verbose=False):
    
    """
    Runs the CPM feature selection step: 
    - correlates each edge with behavior, and returns a mask of edges that are correlated above some threshold, one for each tail (positive and negative)
    """

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    # Correlate all edges with behav vector
    if corr_type =='pearson':
        cov = np.dot(train_behav.T - train_behav.mean(), train_vcts - train_vcts.mean(axis=0)) / (train_behav.shape[0]-1)
        corr = cov / np.sqrt(np.var(train_behav, ddof=1) * np.var(train_vcts, axis=0, ddof=1))
    elif corr_type =='spearman':
        corr = []
        for edge in train_vcts.columns:
            r_val = sp.stats.spearmanr(train_vcts.loc[:,edge], train_behav)[0]
            corr.append(r_val)

    # Define positive and negative masks
    mask_dict = {}
    mask_dict["pos"] = corr > r_thresh
    mask_dict["neg"] = corr < -r_thresh
    
    if verbose:
        print("Found ({}/{}) edges positively/negatively correlated with behavior in the training set".format(mask_dict["pos"].sum(), mask_dict["neg"].sum())) # for debugging

    return mask_dict


def build_model(train_vcts, mask_dict, train_behav):
    """
    Builds a CPM model:
    - takes a feature mask, sums all edges in the mask for each subject, and uses simple linear regression to relate summed network strength to behavior
    """

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    model_dict = {}

    # Loop through pos and neg tails
    X_glm = np.zeros((train_vcts.shape[0], len(mask_dict.items())))

    t = 0
    for tail, mask in mask_dict.items():
        X = train_vcts.values[:, mask].sum(axis=1)
        X_glm[:, t] = X
        y = train_behav
        (slope, intercept) = np.polyfit(X, y, 1)
        model_dict[tail] = (slope, intercept)
        t+=1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    model_dict["glm"] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])

    return model_dict


def apply_model(test_vcts, mask_dict, model_dict):
    """
    Applies a previously trained linear regression model to a test set to generate predictions of behavior.
    """

    behav_pred = {}

    X_glm = np.zeros((test_vcts.shape[0], len(mask_dict.items())))

    # Loop through pos and neg tails
    t = 0
    for tail, mask in mask_dict.items():
        X = test_vcts.loc[:, mask].sum(axis=1)
        X_glm[:, t] = X

        slope, intercept = model_dict[tail]
        behav_pred[tail] = slope*X + intercept
        t+=1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    behav_pred["glm"] = np.dot(X_glm, model_dict["glm"])

    return behav_pred


def cpm_wrapper(all_fc_data, all_behav_data, behav, k=10, **cpm_kwargs):

    assert all_fc_data.index.equals(all_behav_data.index), "Row (subject) indices of FC vcts and behavior don't match!"

    subj_list = all_fc_data.index # get subj_list from df index
    
    indices = mk_kfold_indices(subj_list, k=k)
    
    # Initialize df for storing observed and predicted behavior
    col_list = []
    for tail in ["pos", "neg", "glm"]:
        col_list.append(behav + " predicted (" + tail + ")")
    col_list.append(behav + " observed")
    behav_obs_pred = pd.DataFrame(index=subj_list, columns = col_list)
    
    # Initialize array for storing feature masks
    n_edges = all_fc_data.shape[1]
    all_masks = {}
    all_masks["pos"] = np.zeros((k, n_edges))
    all_masks["neg"] = np.zeros((k, n_edges))
    
    for fold in range(k):
        #print("doing fold {}".format(fold))
        train_subs, test_subs = split_train_test(subj_list, indices, test_fold=fold)
        train_vcts, train_behav, test_vcts = get_train_test_data(all_fc_data, train_subs, test_subs, all_behav_data, behav=behav)
        mask_dict = select_features(train_vcts, train_behav, **cpm_kwargs)
        all_masks["pos"][fold,:] = mask_dict["pos"]
        all_masks["neg"][fold,:] = mask_dict["neg"]
        model_dict = build_model(train_vcts, mask_dict, train_behav)
        behav_pred = apply_model(test_vcts, mask_dict, model_dict)
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, behav + " predicted (" + tail + ")"] = predictions
            
    behav_obs_pred.loc[subj_list, behav + " observed"] = all_behav_data[behav]
    
    return behav_obs_pred, all_masks


def plot_predictions(behav_obs_pred, tail="glm"):
    x = behav_obs_pred.filter(regex=("obs")).astype(float)
    y = behav_obs_pred.filter(regex=(tail)).astype(float)

    g = sns.regplot(x=x.T.squeeze(), y=y.T.squeeze(), color='gray')
    ax_min = min(min(g.get_xlim()), min(g.get_ylim()))
    ax_max = max(max(g.get_xlim()), max(g.get_ylim()))
    g.set_xlim(ax_min, ax_max)
    g.set_ylim(ax_min, ax_max)
    g.set_aspect('equal', adjustable='box')
    
    r = sp.stats.pearsonr(x.values.ravel(),y.values.ravel())[0]
    g.annotate('r = {0:.2f}'.format(r), xy = (0.7, 0.1), xycoords = 'axes fraction')
    
    return g

#%% Create FC based on extracted timeseries
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
from matplotlib import pyplot as plt
import seaborn as sns

tasks = ['REST1', 'REST2', 'WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']

task = tasks[0]

# Load the timeserie, FC (corr), and tril FC
ts = np.load('/Volumes/Elements/HCP_Motion/ts_' + task + '.npy')
# Calculate correlation + fisher r-to-z
# fisher r-to-z transform: to make the values of FC matrix to be of normal distribution
corr = np.arctanh(correlation_measure.fit_transform(ts))
# get the lower triangular
tril = []
for i in range(len(ts)): 
    # lower triangular
    tril.append(corr[i][np.tril_indices(corr.shape[2], k = -1)]) # K = -1 [without diagonal] or 0 [with]
tril = np.array(tril)
fc_all = pd.DataFrame(tril, index=subj_list)

#%% Create FC based on extracted timeseries
from nilearn.connectome import ConnectivityMeasure
from scipy import stats as sp
import numpy as np
import pandas as pd
import pickle

# List of tasks
tasks = ['REST1', 'REST2', 'WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']

# Loop over tasks
for task in tasks:
    print(f"Processing task: {task}")
    
    # Load the timeseries, FC (corr), and tril FC
    ts = np.load('/Volumes/Elements/HCP_Motion/ts_' + task + '.npy')
    
    # Calculate correlation + Fisher r-to-z
    correlation_measure = ConnectivityMeasure(kind='correlation')
    corr = np.arctanh(correlation_measure.fit_transform(ts))
    
    # Get the lower triangular
    tril = []
    for i in range(len(ts)): 
        tril.append(corr[i][np.tril_indices(corr.shape[2], k=-1)]) # K = -1 [without diagonal] or 0 [with]
    tril = np.array(tril)
    fc_all = pd.DataFrame(tril, index=subj_list)
    
    #% Now we run CPM! for two CLUSTERS
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    df['Consensus'] = consensus

    # Choose which behavior you'd like to predict
    behav = 'IQ' # IQ -> TH=0.1

    cpm_kwargs = {'r_thresh': 0.10, 'corr_type': 'pearson'} # these are the defaults, but it's still good to be explicit

    # Define a list of proportions you want to consider (e.g., 0.05, 0.10, 0.15, ..., 1.00)
    proportion_range = [i / 100 for i in range(50, 101, 5)]  # Converts percentages to proportions

    # Create a dictionary to store correlations for each proportion in each cluster
    correlations = {cluster: {proportion: [] for proportion in proportion_range} for cluster in range(2)}

    n_runs = 100

    # Create a pool of subjects with the smallest FD (120 subjects)
    pool_low_movers = df.loc[df['Consensus'] == 0].nsmallest(sum(consensus), 'ALL: FD').index

    # Main loop
    for cluster in range(2):
        for proportion in proportion_range:
            r = []
            n_samples = int(proportion * sum(consensus))  # Calculate the number of samples for the current proportion
            
            for run in range(n_runs):
                # Extract FC for low/high movers based on Consensus and sample size
                if cluster == 0:
                    subj_list_cluster = np.random.choice(pool_low_movers, size=n_samples, replace=False)
                else:
                    subj_list_cluster = df.loc[df['Consensus'] == 1].sample(n_samples, replace=False).index
                
                fc_cluster = fc_all.loc[subj_list_cluster]
                df_cluster = df.loc[subj_list_cluster]
                
                behav_obs_pred, all_masks = cpm_wrapper(fc_cluster, df_cluster, behav=behav, **cpm_kwargs)
                
                x = behav_obs_pred[behav + ' observed']
                y = behav_obs_pred[behav + ' predicted (glm)']
                
                r.append(sp.pearsonr(x, y)[0])  # Calculate and append the Pearson correlation
            
            correlations[cluster][proportion] = np.array(r)  # Store correlations for the current cluster and proportion
            mean_corr = np.mean(r)
            
            print(f"Cluster: {cluster}, Proportion: {proportion:.2f}, Mean Correlation: {mean_corr:.4f}")

    # Save the correlations dictionary to a file for each task
    corr_path = '/Volumes/Elements/HCP_Motion/CPM_corr_' + task + '.pkl'
    with open(corr_path, 'wb') as file:
        pickle.dump(correlations, file)
    
    print(f"Task {task} completed.\n")

#%% Now we run CPM! for two CLUSTERS
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

df['Consensus'] = consensus

# Choose which behavior you'd like to predict
behav = 'IQ' # IQ -> TH=0.1

cpm_kwargs = {'r_thresh': 0.10, 'corr_type': 'pearson'} # these are the defaults, but it's still good to be explicit

# Define a list of proportions you want to consider (e.g., 0.05, 0.10, 0.15, ..., 1.00)
proportion_range = [i / 100 for i in range(50, 101, 5)]  # Converts percentages to proportions

# Create a dictionary to store correlations for each proportion in each cluster
correlations = {cluster: {proportion: [] for proportion in proportion_range} for cluster in range(2)}

n_runs = 100

# Create a pool of subjects with the smallest FD (120 subjects)
pool_low_movers = df.loc[df['Consensus'] == 0].nsmallest(sum(consensus), 'ALL: FD').index

# Main loop
for cluster in range(2):
    for proportion in proportion_range:
        r = []
        n_samples = int(proportion * sum(consensus))  # Calculate the number of samples for the current proportion
        
        for run in range(n_runs):
            # Extract FC for low/high movers based on Consensus and sample size
            if cluster == 0:
                subj_list_cluster = np.random.choice(pool_low_movers, size=n_samples, replace=False)
            else:
                subj_list_cluster = df.loc[df['Consensus'] == 1].sample(n_samples, replace=False).index
            
            fc_cluster = fc_all.loc[subj_list_cluster]
            df_cluster = df.loc[subj_list_cluster]
            
            behav_obs_pred, all_masks = cpm_wrapper(fc_cluster, df_cluster, behav=behav, **cpm_kwargs)
            
            x = behav_obs_pred[behav + ' observed']
            y = behav_obs_pred[behav + ' predicted (glm)']
            
            r.append(sp.stats.pearsonr(x, y)[0])  # Calculate and append the Pearson correlation
        
        correlations[cluster][proportion] = np.array(r)  # Store correlations for the current cluster and proportion
        mean_corr = np.mean(r)
        
        print(f"Cluster: {cluster}, Proportion: {proportion:.2f}, Mean Correlation: {mean_corr:.4f}")

import pickle
# Specify the file path where you want to save/load the correlations
corr_path = '/Volumes/Elements/HCP_Motion/CPM_corr_' + task + '.pkl'
# Save the correlations dictionary to a file
with open(corr_path, 'wb') as file:
    pickle.dump(correlations, file)

#%% plot sample size ratio vs correlations

tasks = ['REST1', 'REST2', 'WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']

corr_path = '/Volumes/Elements/HCP_Motion/CPM_corr_' + 'MOTOR' + '.pkl'

# Load the correlations dictionary from the file
with open(corr_path, 'rb') as file:
    loaded_correlations = pickle.load(file)
    
import matplotlib.pyplot as plt
import numpy as np

# Define the mean correlations for low movers and high movers (replace with your actual data)
mean_correlations_low_movers = np.mean(np.array([loaded_correlations[0][proportion] for proportion in proportion_range]), axis=1)
mean_correlations_high_movers = np.mean(np.array([loaded_correlations[1][proportion] for proportion in proportion_range]), axis=1)

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 5))

# Plot mean correlations for low movers and high movers on the same y-axis
ax.set_xlabel('Sample Size Ratio', fontsize=14)  # Increased font size for x-axis label
ax.set_ylabel('Mean Correlation (Pearson)', color='black', fontsize=14)  # Increased font size for y-axis label
ax.plot(proportion_range, mean_correlations_low_movers, marker='o', linestyle='-', color='#0080FE', label='Low Movers')
ax.plot(proportion_range, mean_correlations_high_movers, marker='o', linestyle='-', color='#BF0A30', label='High Movers')

# Add legend
ax.legend(loc='upper left')

# Set title and grid
plt.title('MOTOR', fontsize=16)
plt.grid(True)

# Show the plot
plt.show()

#%% Boxplot of CPM for CLUSTERS

import numpy as np
import matplotlib.pyplot as plt

# Create a figure with one subplot
fig, ax = plt.subplots(figsize=(4, 6))

# Add the boxplot and jittered data points
ax.boxplot(r_list, showfliers=False, boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2), # r_list[::2]
           capprops=dict(linewidth=2), medianprops=dict(linewidth=2))
for i, r in enumerate(r_list):
    x = np.random.normal(loc=i + 1, scale=0.05, size=len(r))
    ax.scatter(x, r, color='black', alpha=0.5, s=20)

# Set the axis labels and title
ax.set_xlabel('Clusters (Mover Groups)', fontsize=16)
ax.set_ylabel('Correlation', fontsize=16)
ax.set_title(task, fontsize=18)

# Customize the tick labels
ax.set_xticklabels(['C1', 'C2'], fontsize=14)
ax.tick_params(axis='y', labelsize=14)

# Add a horizontal grid
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

# Set the y-axis limits
#ax.set_ylim([-1.0, 1.0])

# Show the plot
plt.show()

#%% Now we run CPM! for QUANTILES
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# Extract FC based on quantiles
condition = 'ALL: FD'
q = df[condition].quantile([0, 0.25, 0.5, 0.75, 1.0])
# Create a dictionary that maps the quantile values to labels
quantile_labels = {q[0.25]: 'Q1', q[0.5]: 'Q2', q[0.75]: 'Q3', q[1.0]: 'Q4'}
# Create a new column 'Motion Group' by mapping the column of interest to quantile labels
df['Motion Group'] = pd.cut(df[condition], bins=q, labels=quantile_labels.values())
df['Motion Group'] = df['Motion Group'].fillna('Q1') # min is Nan sometimes
# Create a KDE plot with different quantiles colored by 'Motion Group'
#chart = sns.displot(data=df, x='BMI', hue='Motion Group', hue_order=quantile_labels.values(), kind='kde', fill=True, palette=sns.color_palette('bright')[:4], height=5, aspect=1.5)
# Obtain the row indices of each Q group
subj_list_q1 = df.loc[df['Motion Group'] == 'Q1'].index
subj_list_q2 = df.loc[df['Motion Group'] == 'Q2'].index
subj_list_q3 = df.loc[df['Motion Group'] == 'Q3'].index
subj_list_q4 = df.loc[df['Motion Group'] == 'Q4'].index
# Get the FC of each motion group (Q1, Q2, Q3, and Q4)
fc_q1 = fc_all.loc[subj_list_q1]
fc_q2 = fc_all.loc[subj_list_q2]
fc_q3 = fc_all.loc[subj_list_q3]
fc_q4 = fc_all.loc[subj_list_q4]
# Get the df of each motion group (Q1, Q2, Q3, and Q4)
df_q1 = df.loc[subj_list_q1]
df_q2 = df.loc[subj_list_q2]
df_q3 = df.loc[subj_list_q3]
df_q4 = df.loc[subj_list_q4]

# Choose which behavior you'd like to predict
behav = 'IQ' # IQ -> TH=0.1
sns.distplot(df_q4[behav])
plt.show()

cpm_kwargs = {'r_thresh': 0.1, 'corr_type': 'pearson'} # these are the defaults, but it's still good to be explicit

fc_list = [fc_q1, fc_q2, fc_q3, fc_q4]
df_list = [df_q1, df_q2, df_q3, df_q4]
r_list = []

n_runs = 100

# main
for quartile in range(4):
    
    r = []
    for run in range(n_runs):
        behav_obs_pred, all_masks = cpm_wrapper(fc_list[quartile], df_list[quartile], 
                                                behav=behav, **cpm_kwargs)
    
        x = behav_obs_pred[behav + ' observed']
        y = behav_obs_pred[behav + ' predicted (glm)']
    
        #r.append(sp.stats.pearsonr(x, y)[0]) # r: correlation
        r.append(spearmanr(x, y)[0]) # r: rank correlation
        #r.append(r2_score(x, y)) # R-squared
        
        print("Run number: {}".format(run))
    
    r = np.array(r)
    print (np.mean(r))
    
    r_list.append(r)

#%% Boxplot of CPM for QUANTILES

import numpy as np
import matplotlib.pyplot as plt

# Create a figure with one subplot
fig, ax = plt.subplots(figsize=(8, 6))

# Add the boxplot and jittered data points
ax.boxplot(r_list, showfliers=False, boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2),
           capprops=dict(linewidth=2), medianprops=dict(linewidth=2))
for i, r in enumerate(r_list):
    x = np.random.normal(loc=i + 1, scale=0.05, size=len(r))
    ax.scatter(x, r, color='black', alpha=0.5, s=20)

# Set the axis labels and title
ax.set_xlabel('Quantiles (Mover Groups)', fontsize=16)
ax.set_ylabel('Correlation', fontsize=16)
ax.set_title(task, fontsize=18)

# Customize the tick labels
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], fontsize=14)
ax.tick_params(axis='y', labelsize=14)

# Add a horizontal grid
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

# Set the y-axis limits
#ax.set_ylim([-1.0, 1.0])

# Show the plot
plt.show()

#%%
# scatter plots of observed versus predicted behavior
g = plot_predictions(behav_obs_pred)
g.set_title(condition)
plt.show()


#%%


x = behav_obs_pred.filter(regex=("obs")).astype(float)
y = behav_obs_pred.filter(regex=("glm")).astype(float)
sns.regplot(x=x.T.squeeze(), y=y.T.squeeze(), color='gray')
r = sp.stats.pearsonr(x.T.squeeze(),y.T.squeeze())
print(r)


















   
#%% REGRESSION ANALYSIS based on FC patterns (lower triangular) 
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

# Building model

X = tril
y = np.array(df.loc[indices, 'Income'])

# Create mask for NaN values in target array
mask = ~np.isnan(y)

# Apply mask to input and target arrays
X_clean = X[mask]
y_clean = y[mask]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Create a linear regression object
reg = LinearRegression()
reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization  

# Fit the model using the training data
reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = reg.predict(X_test)

# Calculate the mean squared error and r2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)










   