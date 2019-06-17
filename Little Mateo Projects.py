import nems.recording as recording
import nems_lbhb.baphy as nb
import nems.preprocessing as preproc
import nems.epoch as nep

import  numpy as np
import scipy.stats as sst
import itertools as itt
import math
import matplotlib.pyplot as plt

'''
first attempt of addapting the old CPP analisie to the new datasets using runclass CPN (context probe natural sound)
CPN contains two main variationse "triplets" and "all permutations".

1. Triplets: the second iteration on CPP, uses adyacent snippets of sounds and shuffles theirs order. the purpose o to
study how different or similar statistics from the context in relation to the probe can influence more or less the probe 
response

2. All permutations: variation on the first iteration of CPP, instead of using artificial vocalization, it uses 4 
different natural sounds
'''

# find cells/site
# CPN
#site = 'AMT031a' # low response, bad
site = 'AMT032a' # great site. PEG
#site = 'ley070a' # good site. A1
#site = 'AMT030a' # low responses, Ok but not as good
#site = 'ley072b' # Primary looking responses with strong contextual effects


modelname = 'resp'
options = {'batch': 316,
           'siteid': site,
           'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
           'runclass': 'CPN',
           'stim': False}  #ToDo chace stims, spectrograms???

load_URI = nb.baphy_load_recording_uri(**options)
loaded_rec = recording.load_recording(load_URI)

##########

rec = loaded_rec
dir(rec)
rec.signals

start = 15
stop = 986


signal = rec['resp']
data = signal.rasterize()._data
data.shape
data[2,1:1000:1]
cell = data[2,start:stop]
cell.shape   # should be 1000
time = np.linspace(start/signal.fs,stop/signal.fs,stop-start)

plt.figure()
plt.plot(time,cell)
plt.title('ley072b-08-1')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate')

signal.chans[2]

first_ep = signal.rasterize().extract_epoch('STIM_sequence002: 6 , 5 , 3 , 2 , 6')
first_ep.shape
(15, 16, 700)

##########

#graph 3rd (2) cell average





first_ep = signal.rasterize().extract_epoch('STIM_sequence002: 6 , 5 , 3 , 2 , 6')
mean = np.mean(first_ep,0) #0 refers to the number of the dimensions, there's 3, so 0 1 2

channel = signal.chans
mycell = ['AMT032a-40-2']
index = channel.index('AMT032a-40-2')
# channel[22]

cell3 = mean[2,0:700]


mean[index]
number3 = mean[index]

plt.figure()
plt.plot(number3)
plt.xlabel('Time (s)')

#    AMT032a-40-2


#######

#Make four 2x2 subplots from cell1, 2, 3, 4

load_URI = nb.baphy_load_recording_uri(**options) #load the file
loaded_rec = recording.load_recording(load_URI)
rec = loaded_rec                        #rename the loaded file
signal = rec['resp']                    #signal is respiration, rename to signal
first_ep = signal.rasterize().extract_epoch('STIM_sequence002: 6 , 5 , 3 , 2 , 6')
    #00001001010 raster, calling that first ep, gets us three dimensions, Rep, cell, time)

PSTHs = np.mean(first_ep,0) #0 refers to the number of the dimensions, there's 3, so 0 1 2

fig,axes = plt.subplots(2,2)   #create 2x2 plot
axes = np.ravel(axes)          #make that into a vector

channel = signal.chans       #naming variable for the cell names

for cells in range(4):
    axes[cells].plot(PSTHs[cells,:])   #index into PSTHs, cell calls the cell, : says all the times
    axes[cells].set_title(channel[cells])   # index single thing because channel is list of names

########

#only contains value of start end and name for name = TRIAL



#epochs.name = epochs['name']    call as method v dictionary
#epochs.iloc[10] ----gives you documention of that number
#create a list with brackets epocs.iloc[[1,2,3,4]]

# == is equal to, not an assignment like =

epochs = signal.epochs
names = epochs["name"]  #names is now the column name, use quotes
istrial = names == "TRIAL"  #boolean as a way of indexing. list of true false

##epochs.loc[row,column] or epochs.loc[[0,1] , ['start','name']]
## can do this with the mask   aaa = epochs.loc[rowmask,colmask]
#i
aaa = epochs.loc[istrial]   ##locate the TRUE

##array only trials and where start time is over 100

start = epochs["start"]
over100 = start > 100

goodtrial = epochs.loc[istrial & over100]    #go through both boolean masks and take TRUE TRUE

#only want to see the "start" column of those ones in good trial

goodtrial = epochs.loc[istrial & over100 , "start"]   #only give specific column of these


##########

