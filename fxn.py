import math
import warnings
import nems.recording as recording
import nems_lbhb.baphy as nb
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

from cpp_parameter_handlers import _epoch_name_handler, _channel_handler, _fs_handler
from nems.signal import PointProcess

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


rec = loaded_rec
signal = rec['resp']                    #signal is response, rename to signal
trials = signal.rasterize().extract_epoch('STIM_sequence002: 6 , 5 , 3 , 2 , 6')
    #00001001010 raster, calling that first ep, gets us three dimensions, Rep, cell, time)

def mateosthing(rec, epoch, first, last):                   #defining inputs
    signal = rec['resp']
    trials = signal.rasterize().extract_epoch(epoch)
    PSTHs = np.mean(trials, 0)                # mean the first dimension of the 3

    axisnum = last-first                         #how many axis based on number of cells

    row = int(np.ceil(math.sqrt(axisnum)))           #make the grid of subplots a square
    col = int(np.floor(math.sqrt(axisnum)))

    if row * col < axisnum:                          #unless it isn't perfect, then add a row
        row = row + 1

    fig, axes = plt.subplots(row, col, sharex=False, sharey=False, squeeze=False)

    axes = np.ravel(axes)

    channel = signal.chans    #variable for cell names

    for cells in range(first,last,1):
        axes[cells].plot(PSTHs[cells,:])
        axes[cells].set_title(channel[cells])

    plt.xlabel('Time')
    plt.ylabel('Firing Rate')

    return fig, axes

mateosthing(rec,'STIM_sequence002: 6 , 5 , 3 , 2 , 6',0,10)