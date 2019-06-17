import math
import warnings
import nems.recording as recording
import nems_lbhb.baphy as nb
import nems_lbhb.io as nio
from nems import epoch as ep

import nems.preprocessing as preproc
import nems.epoch as nep

import  numpy as np
import scipy.stats as sst
import itertools as itt
import math
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.ndimage.filters as sf
import scipy.signal as ssig
import scipy as sp

from cpp_parameter_handlers import _epoch_name_handler, _channel_handler, _fs_handler
from nems.signal import PointProcess

#rec['resp']._data ##calls data that is in signals

mfilename = "/auto/data/daq/Amanita/AMT005/AMT005c05_p_TOR.m"
cellid = 'AMT005c-12-1' #one being used in Matlab
fs=1000
rec = nb.baphy_load_recording_file(mfilename=mfilename, cellid=cellid,fs=fs, stim=False) #fs=1000
globalparams, exptparams, exptevents = nio.baphy_parm_read(mfilename)
signal = rec['resp'].rasterize(fs=fs)                                                         #rasterize the signal


epoch_regex = "^STIM_TORC_.*"                                                            #pick all epochs that have STIM_TORC_...
epochs_to_extract = ep.epoch_names_matching(signal.epochs, epoch_regex)                  #find those epochs
r = signal.extract_epochs(epochs_to_extract)                                             #extract them, r.keys() yields names of TORCS that can be looked through as dic r['name']...can be np.squeeze(0, np.mean(

all_arr = list()                                                                         #create empty list
for val in r.values():                                                                   #for the 30 TORCs in r.values()
    fval = np.swapaxes(np.squeeze(val),0, 1)                                             #create a var to get rid of the third dim (which was a one) and switch the other two axes
    all_arr.append(fval)                                                                 #make all_arr have that swap
stacked = np.stack(all_arr, axis=2)                                                      #stack the #cell on to make like 'r' from MATLAB (time x sweeps x recordings)

#first param in r, time, doesn't match MATLAB, which is 750, here stacked.shape is (145, 10, 30)
TorcObject = exptparams["TrialObject"][1]["ReferenceHandle"][1]                          #will be strf_core_est input
TorcNames = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Names"]
TorcKeys = exptparams["TrialObject"][1]["ReferenceHandle"][1].keys()
RefDuration = TorcObject['Duration']
Events = exptevents


PreStimSil = TorcObject['PreStimSilence']
PostStimSil = TorcObject['PostStimSilence']
#binsize = fs
PreStimbin = int(TorcObject['PreStimSilence']*fs)                                        #how many bins in prestimsilence
PostStimbin = int(TorcObject['PostStimSilence']*fs)                                      #how many bins in poststimsilence
numbin = stacked.shape[0]                                                                #total bins in total time length
INC1stCYCLE = 0                                                                          #default 0
mf = int(fs/1000)                                                                        #silly, but useful later?
stdur = int(RefDuration*1000)
ddur = int(RefDuration*1000)
stonset = 0
stacked = stacked[PreStimbin:(numbin-PostStimbin),:,:]                                   #slice array from first dimensions, bins in pre and post silence, isolate middle 750


###change nesting to TORCs(StimParam(...))
TorcParams = dict.fromkeys(TorcNames)                                                    #Create dict from TorcNames
all_freqs = list()                                                                       #Create empty list of freqs
all_velos = list()                                                                       #Create empty list of velos
all_hfreq = list()
all_lfreq = list()
for tt, torc in enumerate(TorcNames):                                                    #Number them
    TorcParams[torc] = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Params"][tt + 1]     #insert Params 1-30 to torcs 1-30 now TORCs(Params(...)) nested other way
    freqs = TorcParams[torc]['Scales']                                                   #Add all TORCs' Scales value as var
    velos = TorcParams[torc]['Rates']                                                    #Add all TORCs' Rates value as var
    all_freqs.append(freqs)                                                              #
    all_velos.append(velos)                                                              #
    highestfreqs = TorcParams[torc]['HighestFrequency']
    lowestfreqs = TorcParams[torc]['LowestFrequency']
    all_hfreq.append(highestfreqs)
    all_lfreq.append(lowestfreqs)
frqs = np.unique(np.concatenate(all_freqs))                                              #Smoosh to one array and output unique elements
vels = np.unique(np.concatenate(all_velos))                                              #temporal spectra
HighestFrequency = int(np.unique(all_hfreq))
LowestFrequency = int(np.unique(all_lfreq))
Octaves = np.log2(HighestFrequency/LowestFrequency)
phi = 0                                                                                #arbitrarily make 0 per Stephen's code

###
Ompos = [x for x in frqs if x >= 0]                                                      #Get positive frequencies
Omneg = [x for x in frqs if x < 0]                                                       #Get negative frequencies
Omnegzero = [x for x in frqs if x <= 0]                                                  #just used for populating an array a few lines down

N = np.size(frqs) * np.size(vels)        #aka nrips                                      #Seems more concise than MATLAB version --N = size(unique([cat(2,a1rv{:});cat(2,a1rf{:})]','rows'),1)
W = vels                                                                                 #array of ripple velocities
T = np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0])))))               #

Omega = np.swapaxes(np.stack((Ompos,Omnegzero)),0,1)                                     #Make an array for main output Omega

numvels = len(W)                                                                         #
numfrqs = np.size(Omega,0)                                                               #Used to make empty array to be populated by params
numstim = len(TorcNames)

waveParams = np.empty([2,numvels,numfrqs,2,numstim])

##This part in MATLAB makes T, octaves, maxv, maxf, saf, numcomp
basep = np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0])))))
maxvel = np.max(np.abs(W))
maxfrq = np.max(np.abs(Omega))
saf = np.ceil(maxvel*2 + 1000/basep)
numcomp = np.ceil(maxfrq*2*Octaves + 1)

##function [ststims,freqs]=stimprofile(waveParams,W,Omega,lfreq,hfreq,numcomp,T,saf);
[ap,Ws,Omegas,lr,numstim] = waveParams.shape                              #wave params is 5D, define size of each dimension
[a,b] = Omega.shape                                                        #splitting ripple freqs to matrix nums
[d] = W.shape                                                              #splitting W into same

if a*b*d != Omegas*Ws*lr:
    print('Omega and.or W do not match waveParams')

#ln(num)/ln(2) = log2(num)
#numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
freqrange = np.logspace(LowestFrequency, HighestFrequency, numcomp+1, endpoint=True, base=2.0)#make log2 scale from low to high freq binned by how many comps
sffact = saf/1000                                                        #lower sample rate
leng = int(np.round(T*sffact))                                                #stim duration with lower sampling rounded
numcomp = int(numcomp)
t = np.arange(leng)                                                      #make array for time that takes into account 0 is really first

w0 = 1000/T
W0 = 1/(np.log2(HighestFrequency/LowestFrequency))

k = np.round(W/w0)
l = np.round(Omega/W0)

ststims = np.zeros([numcomp, leng, numstim])                            #to be our output of TORCS

cnew = [np.floor(numcomp/2)+1, np.floor(leng/2)+1]

###Function to return the TORC - option to plot each Torc###
def torcmaker(TORC):
    lfreq = TORC['LowestFrequency']
    hfreq = TORC['HighestFrequency']
    Scales = TORC['Scales']
    Amplitude = TORC['RippleAmplitude']
    Phase = TORC['Phase']
    Rate = TORC['Rates']

    octaves = np.log2(hfreq)-np.log2(lfreq)
    normed_scales = [s*octaves for s in Scales]
    cycles_per_sec = 1000/T
    normed_tempmod= [t/cycles_per_sec for t in Rate]

    # somehow we've figured out that final spectrogram should be
    # (numcomp spectral dimension rows) X (leng time samples per TORC cycle)

    stimHolder = np.zeros((numcomp,leng),dtype=complex)
    c = [np.floor(numcomp/2), np.floor(leng/2)]

    for i, (vel,phs,amp,scl) in enumerate(zip(normed_tempmod,Phase,Amplitude,normed_scales)):
        print("ripple {}: vel={}, phi={}, amp={}, scl={}".format(i,vel,phs,amp,scl))

        # figure out index in fourier domain for each ripple
        v1=int(vel+c[1])
        v2=int(c[1]-vel)
        s1=int(scl+c[0])
        s2=int(c[0]-scl)

        stimHolder[s1,v1] = (amp/2)*np.exp(1j*(phs-90)*np.pi/180)
        stimHolder[s2,v2] = (amp/2)*np.exp(-1j*(phs-90)*np.pi/180)
        #print("ripple {}: stimHolder[s1,v1]={}, stimholder[s2,v2]={}".format(i,stimHolder[s1,v1],stimHolder[s2,v2]))
        #######if you want to look at your ripple at any point
        #plt.figure()
        #plt.imshow(np.abs(stimHolder))

    y_sum = (np.fft.ifft2(np.fft.ifftshift(stimHolder*(leng*numcomp)))).real
    return y_sum
######

###this part is important###
TorcValues = dict()                                                                #make new dict for function output
for key,value in TorcParams.items():                                               #cycle through with all the different TORC names and respective values
    y_sum = torcmaker(value)                                                       #output of function (TORC) assigned to variable
    TorcValues[key] = y_sum                                                        #update dict with the key you are on and the value the function just returned



# ######plot all torcs in subplots -!!!eventually make this optional input?
# row = int(np.ceil(math.sqrt(len(TorcValues.keys()))))
# col = int(np.floor(math.sqrt(len(TorcValues.keys()))))
#
# if row * col < len(TorcValues.keys()):
#     row = row + 1
#
# torcfig, torcaxs = plt.subplots(row, col, sharex=False, sharey=False, squeeze=False)
#
# axsall = np.ravel(torcaxs)
#
# plt.xlabel('Time')
# plt.ylabel('Frequency')
#
# for ttt,(key,value) in enumerate(TorcValues.items()):
#     axsall[ttt].imshow(value)
    #axsall[ttt].title[]
#######

########checkmatlab######

# import scipy.io as sio                                             #
# StStims = sio.loadmat('/auto/users/hamersky/stims.mat')            #load Matlab file as dictionary
# ststims = StStims['ststims']                                       #pull out the data as variable
# [mlx,mlt,torcnum] = ststims.shape                                  #assign variables to the shapes
# renametorc = TorcNames                                             #call TorcNames for here
#
# MatlabTorcs = dict()                                               #open new dict to be comparable to TorcValues
# for iii in range(torcnum):                                         #go through in range of 'numstims'
#     whichtorc = ststims[:,:,iii]                                   #each iteration save that torc as variable
#     eee = renametorc[iii]                                          #get the corresponding torcname as num we're on
#     MatlabTorcs[eee] = whichtorc                                   #assign the key and the value, looks like TorcValues

#########################


###########Function to interpolate using FT method, based on matlab interpft()######
def interpft(x,ny,dim):                                                            #input torc array, new sampling, which dimension to interpolate on
    if dim >= 1:                                                                   #if interpolating along columns, dim = 1
        x = np.swapaxes(x,0,dim)                                                   #temporarily swap axes so calculations are universal regardless of dim

    siz = x.shape                                                                  #what is the torc size overall
    [m, n] = x.shape                                                               #unique var for each torc dimension

    if ny > m:                                                                     #if you will be increasing samples (should be)
        incr = 1                                                                   #assign this variable - not very useful but doesn't hurt, could be useful

    a = np.fft.fft(x,m,0)                                                          #do FT along rows, shape unaltered
    nyqst = int(np.ceil((m+1)/2))                                                  #nyqst num calculated
    b = np.concatenate((a[0:nyqst,:], np.zeros(shape=(ny-m,n)), a[nyqst:m, :]),0)  #insert a field of zeros to expand dim to new, using nyqst as break point

    if np.remainder(m,2)==0:                                                       #this hasn't come up yet
        b[nyqst,:] = b[nyqst,:]/2                                                  #presumably dealing with remainder
        b[nyqst+ny-m,:] = b[nyqst,:]                                               #somehow

    y = np.fft.irfft(b,b.shape[0],0)                                                   #take inverse FT (real) using new dimension generated along dim 0 of b

    #if all(np.isreal(x)):                                                          #checks to make sure everything is real
    #    y = y.real                                                                 #it is, don't know when this would come up

    y = y * ny / m                                                                 #necessary conversion...
    #y = y[1:ny:incr,:]                                                             #

    y = np.reshape(y, [y.shape[0],siz[1]])                                             #should preserve shape
    if dim >= 1:                                                                   #as above, if interpolating along columns
        y = np.swapaxes(y,0,dim)                                                   #swap axes back and y will be correct

    return y                                                                       #returned value

####################################################################################


###Going into the stimscale#########################################################
#function stim = stimscale(stim,REGIME,OPTION1,OPTION2,tsiz,xsiz); -our data made moddep 0.9 [] 250 30
# 3)'moddep': Specified modulation depth;
#	OPTION1 is the modulation depth as a fraction of 1. The default is 0.9.
#	OPTION2 is not required.

ModulationDepth = 0.9                                                              #default set in matlab program
base1= 0                                                                           #hard coded in matlab
numtorc = len(TorcValues.keys())                                                   #how many torcs there are, again
xSize = int(np.round(10*numcomp/Octaves))   #30                                    #new x sampling rate
tSize = int(10*saf*basep/1000)              #250                                   #new t sampling rate
ScaledTorcs = dict()                                                                #new dictionary for scaled torcs

for key,value in TorcValues.items():                                               #go through this with all torcs
    [xsiz, tsiz] = value.shape                                                     #basic dims of torc
    temp = value                                                                   #pull out numbers to usable variable

    if xSize != xsiz & tSize != tsiz:                                              #when new sampling rate doesn't equal old vals
        temp1 = interpft(interpft(temp,xSize,0),tSize,1)                           #add points, yielding bigger array

        scl = np.max(np.abs([np.min(np.min(temp1)), np.max(np.max(temp1))]))       #largest |value| is scale factor

        temp2 = base1 + temp*ModulationDepth/scl                                   #transform original torc values with moddep and calc'd scale

    ScaledTorcs[key] = temp2                                                       #populate dictionary with new values
####################################################################################



#Back to code - scaled and all good
scaled_torc_size = temp                                                                      #new section, new var
[stimX,stimT] = scaled_torc_size.shape                                                       #have var for dims of torc
binsize = basep/stimT                                                                        #number of bins

strfest = np.zeros([stimX,stimT])                                                            #empty array same size as torc

##only loop over real (nonNaN) reps, may be diff number of reps for diff torcs
#in my code, 'stacked' is matlab's 'r'
if stacked.shape[0] <= fs/4:                                                                     #multiple cycles
    realrepcount = np.max(np.logical_not(np.isnan(r[1,:,1])).ravel().nonzero())+1            #
else:                                                                                        #go to here, all 10 are real and used
    realrepcount = np.max(np.logical_not(np.isnan(stacked[int(np.round(fs/4))+1,:,1])).ravel().nonzero())+1

#[pp,bb]=fileparts(mfilename)                                           ##don't know what this is for...##

if INC1stCYCLE == 1:                                                               #Some maybe useless code, cause I defaulted var to 0
    if stacked.shape[0]>250:                                                           #
        print('Including 1st TORC Cycle')                                          #Stim starts on 0 cause not excluding
    FirstStimTime = 0                                                              #
elif INC1stCYCLE > 0:                                                              #
    FirstStimTime = INC1stCYCLE                                                    #
else:                                                                              #
    FirstStimTime = basep                                                          #This is what will be happening



##################ENTER SNR FUNCTIONS###############################################
####################################################################################
###SUBFUNCTION - insteadofbin()#####################################################
#inputs spikeperiod (250x30)=resp; basep(250)/stimtime(25)=binsize; mf(1)=mf########
#Downsample spike histogram from resp (resolution mf) to resolution by binsize(ms)##
#"Does by sinc-filtering and downsampling instead of binning," whatever that means##
#function dsum = insteadofbin(resp,binsize,mf);                                   ##


def insteadofbin(resp,binsize,mf=1):                                               #mf is optional, but we'll definitely always have
    if len(resp.shape) >= 2:                                                       #added in jackN phase ot account for Dsum input having one dimension in bython (250x1)
        [spikes,records] = resp.shape                                              #break response into its dimensions
    else:
        resp = np.expand_dims(resp, axis=1)
        [spikes,records] = resp.shape

    outlen = int(spikes / binsize / mf)                                            #

    if outlen - np.floor(outlen/1) * 1 > 1:                                        #basically what matlab fxn mod(x,y)
        print('Non-integer # bins. Result may be distorted')                       #check probably for special circumstances

    outlen = np.round(outlen)                                                      #original comments say "round or ceil or floor?" oh well
    dsum = np.zeros([outlen,records])                                              #empty an array to fill below, going to be output

    for rec in range(records):                                                     #going through all records
        temprec = np.fft.fft(resp[:,rec])                                          #fft for each

        if outlen % 2 == 0:                                                        #if even length, create middle point
            temprec[np.ceil((outlen-1)/2)+1] = np.abs(temprec[np.ceil(outlen-1)/2]+1)

        dsum[:, rec] = np.fft.ifft(np.concatenate((temprec[0:int(np.ceil((outlen - 1) / 2) + 1)], np.conj(
            np.flipud(temprec[1:int(np.floor((outlen - 1) / 2) + 1)]))))).real

    return dsum

####################################################################################


####################################################################################
########going into some SNR#########################################################
#snr = get_snr(spdata, stim, basep, mf, waveParams, alrv)                      #####
#r,StStims,StimParam.basep,StimParam.mf,waveParams,StimParam.a1rv)             #####
#r=stacked, StStims=ScaledTorcs, basep=250, mf =1, waveparams is huge, alrv    #####
def get_snr(spdata,stim,basep,mf,allrates):
    spdata[(np.isnan(spdata)).ravel().nonzero()] = 0                                   #get rid of not numbers
    [numdata,numsweeps,numrecs] = spdata.shape                                         #dims to vars
    [stimfreq, stimtime] = temp.shape                                                  #dims to vars

    invlist = []                                                                       #no clue what this is for actually, it's subsequent uses are commented out

    if spdata.shape[2] != len(ScaledTorcs.keys()):                                     #just a check
        print('Number of records and stimuli are not equal')                           #doubt this'll happen

    # for key, value in ScaledTorcs.items():                                             #pretty unnecessary
    #     [stimfreq,stimtime] = value.shape                                              #a better way to get these I think

    #Response variabiity as a function of frequency per stimulus (-pair)#
    #-------------------------------------------------------------------#
    n = numsweeps * numdata / mf / basep                                                                       #
    tmp = spdata[range(int(np.round(np.floor(n / numsweeps) * mf * basep))),:,:]                               #temp is the same as spdata
    spdata2 = np.reshape(tmp, [int(basep * mf), int(numsweeps * np.floor(n / numsweeps)), numrecs], order='F') #moves dims from 0 axis to 1, new shape 250,30,30
    n = numsweeps * np.floor(n / numsweeps)                                                                    #n should still be same as above

    if n/numsweeps > 1:                                                                                        #it will
        spdata2trim = np.delete(spdata2, np.arange(0,n,n/numsweeps), axis=1)                                   #squish out a third of axis 1

    n = spdata2trim.shape[1]                                                                                   #new axis 1 defines this (2/3 of previous n)

    # if not invlist.shape[0] < 2:
    #     vrf = 1e6 * np.square(stimtime) / np.square(basep) / n * np.square(np.std(
    #         np.fft.fft((spdata2trim[:, :, invlist[0, :]] - spdata2trim[:, :, invlist[1.:]]) / 2, spdata2trim.shape[0], 0),
    #         ddof=1, axis=1))
    # else:
    vrf = 1e6 * np.square(stimtime) / np.square(basep) / n * np.square(                                        #
        np.std(np.fft.fft(spdata2trim, spdata2trim.shape[0], 0), ddof=1, axis=1))                              #

    ##############Response power as a function of frequency##############
    #-------------------------------------------------------------------#
    spikeperiod = np.squeeze(np.mean(spdata2trim,1))                                                           #

    #"downsample spike histogram using insteadofbin()"#
    if basep/stimtime != 1/mf:                                                                                 #
        spikeperiod = insteadofbin(spikeperiod,basep/stimtime,mf)                                              #

    spikeperiod = spikeperiod * 1e3 / basep * stimtime                                                         #

    # if not invlist.shape[0] < 2:
    #     spikeperiod = (spikeperiod[:,invlist[0,:]] - spikeperiod[:,invlist[1,:]])/2

    prf = np.square(np.abs(np.fft.fft(spikeperiod,spikeperiod.shape[0],0)))                                    #

    ######Variability of the STRF estimate (for TORC stimuli)############
    #-------------------------------------------------------------------#
    # if not invlist.shape[0] < 2:
    #     stimpospolarity = invlist[0,:]
    # else:
    stimpospolarity = np.arange(0,stim.shape[2],1).astype(int)                                                 #

    stim2 = stim[:,:,stimpospolarity]                                                                          #

    freqindex = np.swapaxes(np.round(np.abs(allrates) * basep/1000),0,1)                                       #
    freqindx = freqindex[:,stimpospolarity]                                                                    #

    AA = 2 * sum(freqindx > 0 ) / np.mean(np.mean(np.square(stim2),axis=0),axis=0)                             #These are 1/a^2 for each stimulus

    #"Estimate of the total power (ps) and error power (pes) of the STRF"
    pt = 0                                                                                                     #start at zero cause we're going to be adding
    pes = 0                                                                                                    #

    for rec in range(stimpospolarity.shape[0]):                                                                #
        pt = pt + 1 / np.square(stimtime) * AA[rec] * np.sum((2 * prf[freqindx[:,rec].astype(int),rec]))       #keep adding pt through each rep
        pes = pes + 1 / np.square(stimtime) * AA[rec] * np.sum((2 * vrf[freqindx[:,rec].astype(int),rec]))     #keep adding pes through each rep

    snr = pt/pes - 1                                                                                           #calc snr
    return snr
                                                                               #####
                                                                               #####
#got snr#                                                                      #####
####################################################################################
####################################################################################

#Get things ready for snr inputs#
stim = np.stack(list(ScaledTorcs.values()),axis=2)                                 #take my lovely dictionary and smoosh it for now

a1rv = []                                                                          #create empty list for all the Torcs' respective velocities
for key,values in TorcParams.items():                                              #iterate through dict
    torcrate = values['Rates']                                                     #access the rate for each key,value combo
    a1rv.append(torcrate)                                                          #add each time this iterates to a list of lists
allrates = np.asarray(a1rv)                                                        #transform list of list into an array

if stacked.shape[1]>1:
    snr = get_snr(stacked,stim,basep,mf,allrates)
else:
    snr = 0

####################################################################################
#######################MAKEPSTH#####################################################
####################################################################################
#makepsth() [wrapdata,cnorm] = makepsth(dsum,fhist,startime,endtime,mf)#############
##"PSTH: Creates a period histogram according to  period iplied by inputfreq FHIST"#

# dsum: the spike data
# fhist: the frequency for which the histogram is performed
# startime: the start of the histogram data (ms)
# endtime: the end of the histogram data (ms)
# mf: multiplication factor

def makepsth(dsum,fhist,startime,endtime,mf=1):
    if fhist == 0:
        fhist = 1000/(endtime-startime)
    dsum = dsum[:]

    period = int(1000 * (1/fhist) * mf)                                            # in samples
    startime = startime * mf                                                       #      "
    endtime = endtime * mf                                                         #      "

    if endtime > len(dsum):
        endtime = len(dsum)

    fillmax = int(np.ceil(endtime/period) * period)
    if fillmax > endtime:
        dsum[endtime+1:fillmax] = np.nan
        endtime = fillmax

    dsum[:startime] = np.nan
    repcount = int(fillmax / period)
    dsum = np.reshape(dsum[:endtime],(period,repcount),order='F')

    wrapdata = np.nansum(dsum,1)                                                  #get 250 list of how many 1s there were
    cnorm =  np.sum(np.logical_not(np.isnan(dsum)),1)                             #get 250 list of how many 1s were possible

    return wrapdata,cnorm

    ##There's extra code that SVD 'hacked' to allow for includsion of the first TORC
####################################################################################
####################################################################################



##Enter 'if jackN' section - I'm going to assume there are none and follow the if not portion of matlab
jackN = 0
if not jackN:                                                                      #normalize by the num of reps that were presented
    for rep in range(realrepcount):                                                #for this particular record (can be variable for DMS or
        for rec in range(numstim):                                                 #any data set that was interrupted in mid of a rep

            if np.all(stacked[0] <= fs/4):                                                 #
                thisrepcount = sum(np.logical_not(np.isnan(stacked[0,:,rec])))     #this line is direct translation from matlab, wasn't used so can't test actual functionality with numbers, does exact same thing though
            else:
                thisrepcount = sum(sum(np.logical_not(np.isnan(stacked[int(np.round(fs/4)+1):,:,0]))) > 0).astype(int)

            if thisrepcount == 0:
                thisrepcount = 1

            spkdata = stacked[:,rep,rec]

            ###Probably not going to happen in this case
            if fs != 1000:                                                         #this might not be a great translation
                spkdata = sp.signal.resample(stacked,1000)                         #probably not, couldn't find ==resample()

            if len(spkdata) < stdur:
                spkdata = np.concatenate((spkdata,(np.ones((stdur-len(spkdata),1)) * np.nan)))
            ###next part will

            [Dsum,cnorm] = makepsth(spkdata.copy(),int(1000/basep),int(FirstStimTime),stdur,mf)

            #Time to normalize by #cycles. May be variable for TORCs, since they could be shorter than length in exptparams
            Dsum = Dsum / (cnorm + (cnorm == 0))

            if binsize > 1/mf:
                Dsum = insteadofbin(Dsum,binsize,mf)
            Dsum = Dsum * (1000/binsize)                                           #normalization by bin size

            if sum(np.isnan(Dsum)) or not thisrepcount:
                print('NaN STRF: rep {} rec {}'.format(rep,rec))
            else:
                Stim = stim[:,:,rec]
                strftemp = np.zeros(strfest.shape)

            for abc in range(stimX):
                stimrow = Stim[abc,:]
                strftemp[abc,:] = np.fft.ifft(np.conj(np.fft.fft(stimrow)) * np.fft.fft(np.squeeze(Dsum))).real

            strftemp = strftemp / stimT                                                     #normalization
            strftemp = strftemp * (2 * N / np.mean(np.mean(np.square(Stim)))/numstim)       #normalization

            strfest = strfest + strftemp / thisrepcount

strfemp = np.zeros(strfest.shape)
#I've left out the last part of strf_est_core() which has what happens if jackN exists, probably easy to throw in if needed
####################################################################################
####################################################################################
####################################################################################


#strf_est_core() outputs strf0 (strfest), snr, StimParams, strfemp
#back to tor_tuning()

#Make strf torc prediction (it's own tiny function in matlab
pred = np.zeros((stimT,numstim))

for rec in range(numstim):
    for X in range(stimX):
        tr = (np.fft.ifft(np.fft.fft(stim[X,:,rec]) * np.fft.fft(strfest[X,:]))).real

        pred[:,rec] = pred[:,rec] + tr

if INC1stCYCLE:
    FirstStimTime = 0
else:
    FirstStimTime = 250
numreps = stacked.shape[1]
numstims = stacked.shape[2]

stackeduse = stacked[FirstStimTime:,:,:]
cyclesperrep = int(stackeduse.shape[0] / basep)
totalreps = numreps * cyclesperrep

stackeduse = np.reshape(stackeduse, [int(basep),totalreps,numstims], order='F')

##not sure why I'm forcing jacks now, but I'll figure it out
jackcount = 16
jstrf = np.zeros((strfest.shape[0],strfest.shape[1],jackcount))
jackstep = totalreps / jackcount
mm = int(np.round(totalreps / 2))
xc = np.expand_dims(np.zeros(jackcount),axis=1)

for jj in range(jackcount):
    estidx = range(mm) + np.round((jj) * jackstep) + 1
    estidx = (np.remainder(estidx - 1, totalreps)).astype(int)
    validx = (np.setdiff1d(range(totalreps), estidx)).astype(int)
    tr = np.expand_dims(np.nanmean(stackeduse[:,estidx,:], 1),axis=1)
    trval = np.nanmean(stackeduse[:,validx,:],1)





jstrf(:,:, jj)=strf_est_core(tr, TorcObject, rasterfs, 1);

pred = strf_torc_pred(jstrf(:,:, jj), StimParam.StStims);
trval2 = zeros(size(pred));
for ii=1:size(trval, 2),
trval2(:, ii)=resample(trval(:, ii), stimT, StimParam.basep);
end
xc(jj) = xcov(trval2(:), pred(:), 0, 'coeff');
end
linpred = mean(xc);













################
repcount = len(stacked[2])


for rep in range(repcount):
    for rec in range(numstims):
    spkdata = stacked[:,rep,rec]

    ##make psth function
    fhist = 1000/T
    dsum = spkdata[:]
    mf = fs/1000
    Period = 1000*(1/fhist)*mf
    StartTime = T*mf
    EndTime = RefDuration*1000*mf

    fillmax = np.ceil(EndTime/Period)*Period
    repcount = fillmax/Period


    [dsum, cnorm] = makepsth(spkdata, 1000 / StimParam.basep, FirstStimTime, ...
    StimParam.stdur, StimParam.mf);




         spkdata = r(:,rep,rec);
         if rasterfs~=1000,
            spkdata=resample(spkdata,1000,rasterfs);
         end


###In the regular function it generates how to get WaveParams, I think this is where I would switch that because I'm doing the nesting the other way
Joe = TorcParams['TORC_448_LIN_01_h501']

lfreq = Joe['LowestFrequency']
hfreq = Joe['HighestFrequency']
octaves = np.log2(hfreq/lfreq)
RipAmp = Joe['RippleAmplitude']
Scales = Joe['Scales']
Phase = Joe['Phase']
Rate = Joe['Rates']

###Into makestimprofile [waveParams,W,Omega,N,T,k] = makestimprofile(a1am=RipAmp,a1rf=Scales,a1ph=Phase,a1rv=Rate)

numstims = len(TorcParams)                                                                    #TORCs in dict says how many total as ref
vels = Rate
for freqs in enumerate(TorcParams.keys()):



frqs = unique(exptparams.TORCs[])


vels = cat(2,a1rv{:});
frqs = cat(2,a1rf{:});
frqs(find(vels<0)) = -frqs(find(vels<0));
vels = unique(abs(vels));
frqs = np.linspace(-1.400,1.400,15)



N = size(unique([cat(2,a1rv{:});cat(2,a1rf{:})]','rows'),1);
W = vels;
T = round(1000/min(abs(diff([0 unique(abs(W(find(W))))]))));
%k = round(abs(cat(1,a1rv{:}))'*T/1000 + 1);

#########
INC1stCYCLE = 0

referencecount = exptparams["TrialObject"][1]["ReferenceHandle"][1]["MaxIndex"]          #How many TORCs, reference
referenceduration = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Duration"]       #Duration of TORC, reference
##Time to define some parameters for later
StimParam = dict()                                                                       #Make a dictionary to fill down there
StimParam['numrecs'] = referencecount
StimParam['mf'] = fs/1000
StimParam['ddur'] = np.round(1000*referenceduration)
StimParam['stdur'] = np.round(1000*referenceduration)
StimParam['stonset'] = 0
StimParam['lfreq'] = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Params"][1]["LowestFrequency"]
StimParam['hfreq'] = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Params"][1]["HighestFrequency"]
StimParam['octaves'] = np.log2(StimParam['hfreq']/StimParam['lfreq'])
StimParam['alam'] = list()
StimParam['alrf'] = list()
StimParam['alph'] = {}
StimParam['alrv'] = {}

for ii in range(StimParam['numrecs']):
    StimParam['alam'] = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Params"][1]


for ii = 1:StimParam.numrecs
    StimParam['alam'][ii] = TorcObject.Params(ii).RippleAmplitude;
    StimParam.a1rf{ii} = TorcObject.Params(ii).Scales;
    StimParam.a1ph{ii} = TorcObject.Params(ii).Phase;
    StimParam.a1rv{ii} = TorcObject.Params(ii).Rates;
end;

###[strf0,snr,StimParam,strfee]=strf_est_core(r,TorcObject,fs,INC1stCYCLE,16);








###########################################
##this may be useful for poststimsilence stuff going up to where r is def
    d = signal.get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d)) - 0.5/fs
    dur = signal.get_epoch_bounds(epochs_to_extract[0])
    FullDuration = np.mean(np.diff(dur)) - 0.5/fs
    d = signal.get_epoch_bounds('PostStimSilence')
    if d.size > 0:
        PostStimSilence = np.min(np.diff(d)) - 0.5/fs
        dd = np.diff(d)
        dd = dd[dd > 0]
    else:
        dd = np.array([])
    if dd.size > 0:
        PostStimSilence = np.min(dd) - 0.5/fs
    else:
        PostStimSilence = 0
    Duration = FullDuration - PostStimSilence - PreStimSilence

    r = signal.extract_epochs(epochs_to_extract)

##########################################


#r = time x numsweeps x nrecordings




[strf0,snr,StimParam,strfee]=strf_est_core(r,TorcObject,fs,INC1stCYCLE,16);


####
####
####





###
#Make TORCS

function [ststims,freqs]=stimprofile(waveParams,W,Omega,...                 #all of this lives in TorcObject (what is W)
				 lfreq,hfreq,numcomp,T,saf);

def stimprofile(waveParams,W,Omega,lfreq,hfreq,numcomp,T,saf=1000):             #=1000 sets default to 1000 if not inputted, like nargin


#[ststims,freqs] = ststruct(waveParams,W,Omega,lfreq,hfreq,numcomp,T,saf);
#
#STSTRUCT : Spectro-temporal stimulus constructor.
#
#waveParams : 5-D matrix containing amplitude and phase pairs for
#             all components of all waveforms (see MAKEWAVEPARAMS)
#W          : Vector of ripple velocities
#Omega      : Matrix of ripple frequencies
#lfreq      : lowest frequency component
#hfreq      : highest frequency component
#numcomp    : number of components (frequency length)
#T          : stimulus duration (ms)
#saf 	     : sampling frequency (default = 1000 Hz)

          %

###
###

def tor_tuning(mfilename,channum,unit,plotaxes):
    load_URI = nb.baphy_load_recording_uri(**options)
    loaded_rec = recording.load_recording(load_URI)
    rec = loaded_rec

    signal = rec['resp']
    signals = rec.signals

    parmfile =




################Old version torcmaker() I was working with that didn't quite work
    # t = np.expand_dims(np.linspace(lrate,hrate,leng, False), axis=0)
    # f = np.expand_dims(np.linspace(lfreq,hfreq,numcomp, False), axis=1)
    #
    # # fig, axs = plt.subplots(1, len(Rate) + 1,squeeze=False)
    # # im_extent=[0,np.max(t),0,np.max(f)]
    # y_sum = np.zeros((t + f).shape)
    #
    # import pdb
    # pdb.set_trace()
    #
    # for i, (vel,phi,amp,scl) in enumerate(zip(Rate,Phase,Amplitude,Scales)):
    #     y = (amp) * (np.sin(vel * t + scl * f + phi))
    #
    #     y_sum += y

    #     axs[0, i].imshow(y, aspect='auto', origin='lower', extent=im_extent)
    #     if (TorcFreq < np.max(frqs)) or (i > 0):
    #         axs[0,i].get_xaxis().set_visible(False)
    #     else:
    #         axs[0, i].tick_params(labelsize=6)
    #     axs[0, i].set_yticks([])
    #     if i==0:
    #         axs[0, i].set_ylabel("{:.1f} cyc/oct".format(scl), fontsize=6)
    #
    #     #plot the summed TORC
    # axs[0, i+1].imshow(y_sum, aspect='auto', origin='lower', extent=im_extent)
    # axs[0, i+1].get_xaxis().set_visible(False)
    # axs[0, i+1].get_yaxis().set_visible(False)
    #
    # fig1, axs1 = plt.subplots()
    # axs1.imshow(y_sum, aspect='auto', origin='lower', extent=im_extent)

    # return y_sum
############################

